//===--- APINotesMAnager.cpp - Manage API Notes Files ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the APINotesManager class.
//
//===----------------------------------------------------------------------===//

#include "clang/APINotes/APINotesManager.h"
#include "clang/APINotes/APINotesOptions.h"
#include "clang/APINotes/APINotesReader.h"
#include "clang/APINotes/APINotesYAMLCompiler.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceMgrAdapter.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Path.h"
#include <sys/stat.h>

using namespace clang;
using namespace api_notes;

#define DEBUG_TYPE "API Notes"
STATISTIC(NumHeaderAPINotes,
          "non-framework API notes files loaded");
STATISTIC(NumPublicFrameworkAPINotes,
          "framework public API notes loaded");
STATISTIC(NumPrivateFrameworkAPINotes,
          "framework private API notes loaded");
STATISTIC(NumFrameworksSearched,
          "frameworks searched");
STATISTIC(NumDirectoriesSearched,
          "header directories searched");
STATISTIC(NumDirectoryCacheHits,
          "directory cache hits");
STATISTIC(NumBinaryCacheHits,
          "binary form cache hits");
STATISTIC(NumBinaryCacheMisses,
          "binary form cache misses");
STATISTIC(NumBinaryCacheRebuilds,
          "binary form cache rebuilds");

APINotesManager::APINotesManager(SourceManager &sourceMgr,
                                 const LangOptions &langOpts)
  : SourceMgr(sourceMgr), ImplicitAPINotes(langOpts.APINotes),
    PrunedCache(false) { }

APINotesManager::~APINotesManager() {
  // Free the API notes readers.
  for (const auto &entry : Readers) {
    if (auto reader = entry.second.dyn_cast<APINotesReader *>()) {
      delete reader;
    }
  }
}

/// \brief Write a new timestamp file with the given path.
static void writeTimestampFile(StringRef TimestampFile) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(TimestampFile.str(), EC, llvm::sys::fs::F_None);
}

/// \brief Prune the API notes cache of API notes that haven't been accessed in
/// a long time.
static void pruneAPINotesCache(StringRef APINotesCachePath) {
  struct stat StatBuf;
  llvm::SmallString<128> TimestampFile;
  TimestampFile = APINotesCachePath;
  llvm::sys::path::append(TimestampFile, "APINotes.timestamp");

  // Try to stat() the timestamp file.
  if (::stat(TimestampFile.c_str(), &StatBuf)) {
    // If the timestamp file wasn't there, create one now.
    if (errno == ENOENT) {
      llvm::sys::fs::create_directories(APINotesCachePath);
      writeTimestampFile(TimestampFile);
    }
    return;
  }

  const unsigned APINotesCachePruneInterval = 7 * 24 * 60 * 60;
  const unsigned APINotesCachePruneAfter = 31 * 24 * 60 * 60;

  // Check whether the time stamp is older than our pruning interval.
  // If not, do nothing.
  time_t TimeStampModTime = StatBuf.st_mtime;
  time_t CurrentTime = time(nullptr);
  if (CurrentTime - TimeStampModTime <= time_t(APINotesCachePruneInterval))
    return;

  // Write a new timestamp file so that nobody else attempts to prune.
  // There is a benign race condition here, if two Clang instances happen to
  // notice at the same time that the timestamp is out-of-date.
  writeTimestampFile(TimestampFile);

  // Walk the entire API notes cache, looking for unused compiled API notes.
  std::error_code EC;
  SmallString<128> APINotesCachePathNative;
  llvm::sys::path::native(APINotesCachePath, APINotesCachePathNative);
  for (llvm::sys::fs::directory_iterator
         File(APINotesCachePathNative.str(), EC), DirEnd;
       File != DirEnd && !EC; File.increment(EC)) {
    StringRef Extension = llvm::sys::path::extension(File->path());
    if (Extension.empty())
      continue;

    if (Extension.substr(1) != BINARY_APINOTES_EXTENSION)
      continue;

    // Look at this file. If we can't stat it, there's nothing interesting
    // there.
    if (::stat(File->path().c_str(), &StatBuf))
      continue;

    // If the file has been used recently enough, leave it there.
    time_t FileAccessTime = StatBuf.st_atime;
    if (CurrentTime - FileAccessTime <= time_t(APINotesCachePruneAfter)) {
      continue;
    }

    // Remove the file.
    llvm::sys::fs::remove(File->path());
  }
}

std::unique_ptr<APINotesReader>
APINotesManager::loadAPINotes(const FileEntry *apiNotesFile) {
  FileManager &fileMgr = SourceMgr.getFileManager();

  // If the API notes file is already in the binary form, load it directly.
  StringRef apiNotesFileName = apiNotesFile->getName();
  StringRef apiNotesFileExt = llvm::sys::path::extension(apiNotesFileName);
  if (!apiNotesFileExt.empty() &&
      apiNotesFileExt.substr(1) == BINARY_APINOTES_EXTENSION) {
    // Load the file.
    auto buffer = fileMgr.getBufferForFile(apiNotesFile);
    if (!buffer) return nullptr;

    // Load the binary form.
    return APINotesReader::get(std::move(buffer.get()));
  }

  // If we haven't pruned the API notes cache yet during this execution, do
  // so now.
  if (!PrunedCache) {
    pruneAPINotesCache(fileMgr.getFileSystemOpts().APINotesCachePath);
    PrunedCache = true;
  }

  // Compute a hash of the API notes file's directory and the Clang version,
  // to be used as part of the filename for the cached binary copy.
  auto code = llvm::hash_value(StringRef(apiNotesFile->getDir()->getName()));
  code = hash_combine(code, getClangFullRepositoryVersion());

  // Determine the file name for the cached binary form.
  SmallString<128> compiledFileName;
  compiledFileName += fileMgr.getFileSystemOpts().APINotesCachePath;
  assert(!compiledFileName.empty() && "No API notes cache path provided?");
  llvm::sys::path::append(compiledFileName,
    (llvm::Twine(llvm::sys::path::stem(apiNotesFileName)) + "-"
     + llvm::APInt(64, code).toString(36, /*Signed=*/false) + "."
     + BINARY_APINOTES_EXTENSION));

  // Try to open the cached binary form.
  if (const FileEntry *compiledFile = fileMgr.getFile(compiledFileName,
                                                      /*openFile=*/true,
                                                      /*cacheFailure=*/false)) {
    // Load the file contents.
    if (auto buffer = fileMgr.getBufferForFile(compiledFile)) {
      // Make sure the file is up-to-date.
      if (compiledFile->getModificationTime()
            >= apiNotesFile->getModificationTime()) {
        // Load the file.
        if (auto reader = APINotesReader::get(std::move(buffer.get()))) {
          // Success.
          ++NumBinaryCacheHits;
          return reader;
        }
      }
    }

    // The cache entry was somehow broken; delete this one so we can build a
    // new one below.
    llvm::sys::fs::remove(compiledFileName.str());
    ++NumBinaryCacheRebuilds;
  } else {
    ++NumBinaryCacheMisses;
  }

  // Open the source file.
  auto buffer = fileMgr.getBufferForFile(apiNotesFile);
  if (!buffer) return nullptr;

  // Compile the API notes source into a buffer.
  // FIXME: Either propagate OSType through or, better yet, improve the binary
  // APINotes format to maintain complete availability information.
  llvm::SmallVector<char, 1024> apiNotesBuffer;
  {
    SourceMgrAdapter srcMgrAdapter(SourceMgr, SourceMgr.getDiagnostics(),
                                   diag::err_apinotes_message,
                                   diag::warn_apinotes_message,
                                   diag::note_apinotes_message,
                                   apiNotesFile);
    llvm::raw_svector_ostream OS(apiNotesBuffer);
    if (api_notes::compileAPINotes(buffer.get()->getBuffer(),
                                   OS,
                                   api_notes::OSType::Absent,
                                   srcMgrAdapter.getDiagHandler(),
                                   srcMgrAdapter.getDiagContext()))
      return nullptr;

    // Make a copy of the compiled form into the buffer.
    buffer = llvm::MemoryBuffer::getMemBufferCopy(
               StringRef(apiNotesBuffer.data(), apiNotesBuffer.size()));
  }

  // Save the binary form into the cache. Perform this operation
  // atomically.
  SmallString<64> temporaryBinaryFileName = compiledFileName.str();
  temporaryBinaryFileName.erase(
    temporaryBinaryFileName.end()
      - llvm::sys::path::extension(temporaryBinaryFileName).size(),
    temporaryBinaryFileName.end());
  temporaryBinaryFileName += "-%%%%%%.";
  temporaryBinaryFileName += BINARY_APINOTES_EXTENSION;

  int temporaryFD;
  llvm::sys::fs::create_directories(
    fileMgr.getFileSystemOpts().APINotesCachePath);
  if (!llvm::sys::fs::createUniqueFile(temporaryBinaryFileName.str(),
                                       temporaryFD, temporaryBinaryFileName)) {
    // Write the contents of the buffer.
    bool hadError;
    {
      llvm::raw_fd_ostream out(temporaryFD, /*shouldClose=*/true);
      out.write(buffer.get()->getBufferStart(), buffer.get()->getBufferSize());
      out.flush();

      hadError = out.has_error();
    }

    if (!hadError) {
      // Rename the temporary file to the actual compiled file.
      llvm::sys::fs::rename(temporaryBinaryFileName.str(),
                            compiledFileName.str());
    }
  }

  // Load the binary form we just compiled.
  auto reader = APINotesReader::get(std::move(*buffer));
  assert(reader && "Could not load the API notes we just generated?");
  return reader;
}

bool APINotesManager::loadAPINotes(const DirectoryEntry *HeaderDir,
                                   const FileEntry *APINotesFile) {
  assert(Readers.find(HeaderDir) == Readers.end());
  if (auto reader = loadAPINotes(APINotesFile)) {
    Readers[HeaderDir] = reader.release();
    return false;
  }

  Readers[HeaderDir] = nullptr;
  return true;
}

const DirectoryEntry *APINotesManager::loadFrameworkAPINotes(
                        llvm::StringRef FrameworkPath,
                        llvm::StringRef FrameworkName,
                        bool Public) {
  FileManager &FileMgr = SourceMgr.getFileManager();
  
  llvm::SmallString<128> Path;
  Path += FrameworkPath;
  unsigned FrameworkNameLength = Path.size();

  // Form the path to the APINotes file.
  llvm::sys::path::append(Path, "APINotes");
  if (Public)
    llvm::sys::path::append(Path,
                            (llvm::Twine(FrameworkName) + "."
                              + SOURCE_APINOTES_EXTENSION));
  else
    llvm::sys::path::append(Path,
                            (llvm::Twine(FrameworkName) + "_private."
                              + SOURCE_APINOTES_EXTENSION));

  // Try to open the APINotes file.
  const FileEntry *APINotesFile = FileMgr.getFile(Path);
  if (!APINotesFile)
    return nullptr;

  // Form the path to the corresponding header directory.
  Path.resize(FrameworkNameLength);
  if (Public)
    llvm::sys::path::append(Path, "Headers");
  else
    llvm::sys::path::append(Path, "PrivateHeaders");

  // Try to access the header directory.
  const DirectoryEntry *HeaderDir = FileMgr.getDirectory(Path);
  if (!HeaderDir)
    return nullptr;

  // Try to load the API notes.
  if (loadAPINotes(HeaderDir, APINotesFile))
    return nullptr;

  // Success: return the header directory.
  if (Public)
    ++NumPublicFrameworkAPINotes;
  else
    ++NumPrivateFrameworkAPINotes;
  return HeaderDir;
}

const FileEntry *APINotesManager::loadCurrentModuleAPINotes(
                   StringRef moduleName,
                   ArrayRef<std::string> searchPaths) {
  assert(!CurrentModuleReader &&
         "Already loaded API notes for the current module?");

  FileManager &fileMgr = SourceMgr.getFileManager();

  // Look for API notes for this module in the module search paths.
  for (const auto &searchPath : searchPaths) {
    // First, look for a binary API notes file.
    llvm::SmallString<128> apiNotesFilePath;
    apiNotesFilePath += searchPath;
    llvm::sys::path::append(
      apiNotesFilePath,
      llvm::Twine(moduleName) + "." + BINARY_APINOTES_EXTENSION);

    // Try to open the binary API Notes file.
    if (const FileEntry *binaryAPINotesFile
          = fileMgr.getFile(apiNotesFilePath)) {
      CurrentModuleReader = loadAPINotes(binaryAPINotesFile);
      return CurrentModuleReader ? binaryAPINotesFile : nullptr;
    }

    // Try to open the source API Notes file.
    apiNotesFilePath = searchPath;
    llvm::sys::path::append(
      apiNotesFilePath,
      llvm::Twine(moduleName) + "." + SOURCE_APINOTES_EXTENSION);
    if (const FileEntry *sourceAPINotesFile
          = fileMgr.getFile(apiNotesFilePath)) {
      CurrentModuleReader = loadAPINotes(sourceAPINotesFile);
      return CurrentModuleReader ? sourceAPINotesFile : nullptr;
    }
  }

  // Didn't find any API notes.
  return nullptr;
}

APINotesReader *APINotesManager::findAPINotes(SourceLocation Loc) {
  // If there is a reader for the current module, return it.
  if (CurrentModuleReader) return CurrentModuleReader.get();

  // If we're not allowed to implicitly load API notes files, we're done.
  if (!ImplicitAPINotes) return nullptr;

  // If we don't have source location information, we're done.
  if (Loc.isInvalid()) return nullptr;

  // API notes are associated with the expansion location. Retrieve the
  // file for this location.
  SourceLocation ExpansionLoc = SourceMgr.getExpansionLoc(Loc);
  FileID ID = SourceMgr.getFileID(ExpansionLoc);
  if (ID.isInvalid())
    return nullptr;
  const FileEntry *File = SourceMgr.getFileEntryForID(ID);
  if (!File)
    return nullptr;

  // Look for API notes in the directory corresponding to this file, or one of
  // its its parent directories.
  const DirectoryEntry *Dir = File->getDir();
  FileManager &FileMgr = SourceMgr.getFileManager();
  llvm::SetVector<const DirectoryEntry *,
                  SmallVector<const DirectoryEntry *, 4>,
                  llvm::SmallPtrSet<const DirectoryEntry *, 4>> DirsVisited;
  APINotesReader *Result = nullptr;
  do {
    // Look for an API notes reader for this header search directory.
    auto Known = Readers.find(Dir);

    // If we already know the answer, chase it.
    if (Known != Readers.end()) {
      ++NumDirectoryCacheHits;

      // We've been redirected to another directory for answers. Follow it.
      if (auto OtherDir = Known->second.dyn_cast<const DirectoryEntry *>()) {
        DirsVisited.insert(Dir);
        Dir = OtherDir;
        continue;
      }

      // We have the answer.
      Result = Known->second.dyn_cast<APINotesReader *>();
      break;
    }

    // Look for API notes corresponding to this directory.
    StringRef Path = Dir->getName();
    if (llvm::sys::path::extension(Path) == ".framework") {
      // If this is a framework directory, check whether there are API notes
      // in the APINotes subdirectory.
      auto FrameworkName = llvm::sys::path::stem(Path);
      ++NumFrameworksSearched;

      // Look for API notes for both the public and private headers.
      const DirectoryEntry *PublicDir
        = loadFrameworkAPINotes(Path, FrameworkName, /*Public=*/true);
      const DirectoryEntry *PrivateDir
        = loadFrameworkAPINotes(Path, FrameworkName, /*Public=*/false);

      if (PublicDir || PrivateDir) {
        // We found API notes: don't ever look past the framework directory.
        Readers[Dir] = nullptr;

        // Pretend we found the result in the public or private directory,
        // as appropriate. All headers should be in one of those two places,
        // but be defensive here.
        if (!DirsVisited.empty()) {
          if (DirsVisited.back() == PublicDir) {
            DirsVisited.pop_back();
            Dir = PublicDir;
          } else if (DirsVisited.back() == PrivateDir) {
            DirsVisited.pop_back();
            Dir = PrivateDir;
          }
        }

        // Grab the result.
        Result = Readers[Dir].dyn_cast<APINotesReader *>();;
        break;
      }
    } else {
      // Look for an APINotes file in this directory.
      llvm::SmallString<128> APINotesPath;
      APINotesPath += Dir->getName();
      llvm::sys::path::append(APINotesPath,
                              (llvm::Twine("APINotes.")
                                 + SOURCE_APINOTES_EXTENSION));

      // If there is an API notes file here, try to load it.
      ++NumDirectoriesSearched;
      if (const FileEntry *APINotesFile = FileMgr.getFile(APINotesPath)) {
        if (!loadAPINotes(Dir, APINotesFile)) {
          ++NumHeaderAPINotes;
          Result = Readers[Dir].dyn_cast<APINotesReader *>();
          break;
        }
      }
    }

    // We didn't find anything. Look at the parent directory.
    if (!DirsVisited.insert(Dir)) {
      Dir = 0;
      break;
    }

    StringRef ParentPath = llvm::sys::path::parent_path(Path);
    while (llvm::sys::path::stem(ParentPath) == "..") {
      ParentPath = llvm::sys::path::parent_path(ParentPath);
    }
    if (ParentPath.empty()) {
      Dir = nullptr;
    } else {
      Dir = FileMgr.getDirectory(ParentPath);
    }
  } while (Dir);

  // Path compression for all of the directories we visited, redirecting
  // them to the directory we ended on. If no API notes were found, the
  // resulting directory will be NULL, indicating no API notes.
  for (const auto Visited : DirsVisited) {
    Readers[Visited] = Dir;
  }

  return Result;
}
