//===--- FileSystemStatCache.cpp - Caching for 'stat' calls ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the FileSystemStatCache interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/FileSystemStatCache.h"
#include "llvm/Support/Path.h"
#include <fcntl.h>

// FIXME: This is terrible, we need this for ::close.
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#include <sys/uio.h>
#else
#include <io.h>
#endif
using namespace clang;

#if defined(_MSC_VER)
#define S_ISDIR(s) ((_S_IFDIR & s) !=0)
#endif

void FileSystemStatCache::anchor() { }

/// FileSystemStatCache::get - Get the 'stat' information for the specified
/// path, using the cache to accelerate it if possible.  This returns true if
/// the path does not exist or false if it exists.
///
/// If FileDescriptor is non-null, then this lookup should only return success
/// for files (not directories).  If it is null this lookup should only return
/// success for directories (not files).  On a successful file lookup, the
/// implementation can optionally fill in FileDescriptor with a valid
/// descriptor and the client guarantees that it will close it.
bool FileSystemStatCache::get(const char *Path, struct stat &StatBuf,
                              int *FileDescriptor, FileSystemStatCache *Cache) {
  LookupResult R;
  bool isForDir = FileDescriptor == 0;

  // If we have a cache, use it to resolve the stat query.
  if (Cache)
    R = Cache->getStat(Path, StatBuf, FileDescriptor);
  else if (isForDir) {
    // If this is a directory and we have no cache, just go to the file system.
    R = ::stat(Path, &StatBuf) != 0 ? CacheMissing : CacheExists;
  } else {
    // Otherwise, we have to go to the filesystem.  We can always just use
    // 'stat' here, but (for files) the client is asking whether the file exists
    // because it wants to turn around and *open* it.  It is more efficient to
    // do "open+fstat" on success than it is to do "stat+open".
    //
    // Because of this, check to see if the file exists with 'open'.  If the
    // open succeeds, use fstat to get the stat info.
    int OpenFlags = O_RDONLY;
#ifdef O_BINARY
    OpenFlags |= O_BINARY;  // Open input file in binary mode on win32.
#endif
    *FileDescriptor = ::open(Path, OpenFlags);
    
    if (*FileDescriptor == -1) {
      // If the open fails, our "stat" fails.
      R = CacheMissing;
    } else {
      // Otherwise, the open succeeded.  Do an fstat to get the information
      // about the file.  We'll end up returning the open file descriptor to the
      // client to do what they please with it.
      if (::fstat(*FileDescriptor, &StatBuf) == 0)
        R = CacheExists;
      else {
        // fstat rarely fails.  If it does, claim the initial open didn't
        // succeed.
        R = CacheMissing;
        ::close(*FileDescriptor);
        *FileDescriptor = -1;
      }
    }
  }

  // If the path doesn't exist, return failure.
  if (R == CacheMissing) return true;
  
  // If the path exists, make sure that its "directoryness" matches the clients
  // demands.
  if (S_ISDIR(StatBuf.st_mode) != isForDir) {
    // If not, close the file if opened.
    if (FileDescriptor && *FileDescriptor != -1) {
      ::close(*FileDescriptor);
      *FileDescriptor = -1;
    }
    
    return true;
  }
  
  return false;
}


MemorizeStatCalls::LookupResult
MemorizeStatCalls::getStat(const char *Path, struct stat &StatBuf,
                           int *FileDescriptor) {
  LookupResult Result = statChained(Path, StatBuf, FileDescriptor);
  
  // Do not cache failed stats, it is easy to construct common inconsistent
  // situations if we do, and they are not important for PCH performance (which
  // currently only needs the stats to construct the initial FileManager
  // entries).
  if (Result == CacheMissing)
    return Result;
  
  // Cache file 'stat' results and directories with absolutely paths.
  if (!S_ISDIR(StatBuf.st_mode) || llvm::sys::path::is_absolute(Path))
    StatCalls[Path] = StatBuf;
  
  return Result;
}
