//
// https://man7.org/linux/man-pages/man3/ftw.3.html
//

/*
 * -----
 * Build
 * -----
 * cd ..../google3/experimental/users/rtenneti/llvm_libc/
 * make
 *
 * ------------
 * How to Test:
 * ------------
 * make test
 */

#include "nftw.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>  // size_t
#include <stdint.h>  // int8_t, int16_t and int32_t
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>  // NOLINT
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>

namespace {
class Func {
 public:
  virtual int call(const char*, const struct stat*, int, struct FTW*) = 0;
};

using nftwFn = int (*)(const char* filePath, const struct stat* statBuf,
                       int tFlag, struct FTW* ftwbuf);

using ftwFn = int (*)(const char* filePath, const struct stat* statBuf,
                      int tFlag);

class NftwFunc : public Func {
 public:
  NftwFunc(nftwFn fn) : fn(fn) {}
  virtual int call(const char* dirPath, const struct stat* statBuf, int tFlag,
                   struct FTW* ftwBuf) override {
    return fn(dirPath, statBuf, tFlag, ftwBuf);
  }

 private:
  const nftwFn fn;
};

class FtwFunc : public Func {
 public:
  FtwFunc(ftwFn fn) : fn(fn) {}
  virtual int call(const char* dirPath, const struct stat* statBuf, int tFlag,
                   struct FTW*) override {
    return fn(dirPath, statBuf, tFlag);
  }

 private:
  const ftwFn fn;
};

int doMergedFtw(const std::string& dirPath, Func& fn, int fdLimit, int flags,
                int level) {
  // fdLimit specifies the maximum number of directories that ftw()
  // will hold open simultaneously. When a directory is opened, fdLimit is
  // decreased and if it becomes 0 or less, we won't open any more directories.
  if (fdLimit <= 0) {
    return 0;
  }

  // Determine the type of path that is passed.
  int typeFlag;
  struct stat statBuf;
  if (flags & FTW_PHYS) {
    if (lstat(dirPath.c_str(), &statBuf) < 0) return -1;
  } else {
    if (stat(dirPath.c_str(), &statBuf) < 0) {
      if (!lstat(dirPath.c_str(), &statBuf)) {
        typeFlag = FTW_SLN; /* Symbolic link pointing to a nonexistent file. */
      } else if (errno != EACCES) {
        /* stat failed with an errror that is not Permission denied */
        return -1;
      } else {
        /* The probable cause for the failure is that the caller had read
         * permission on  the parent directory, so that the filename fpath could
         * be seen, but did not have execute permission on the directory.
         */
        typeFlag = FTW_NS;
      }
    }
  }

  if (S_ISDIR(statBuf.st_mode)) {
    if (flags & FTW_DEPTH) {
      typeFlag = FTW_DP; /* Directory, all subdirs have been visited. */
    } else {
      typeFlag = FTW_D; /* Directory. */
    }
  } else if (S_ISLNK(statBuf.st_mode)) {
    if (flags & FTW_PHYS) {
      typeFlag = FTW_SL; /* Symbolic link.  */
    } else {
      typeFlag = FTW_SLN; /* Symbolic link pointing to a nonexistent file. */
    }
  } else {
    typeFlag = FTW_F; /* Regular file.  */
  }

  struct FTW ftwBuf;
  // Find the base by finding the last slash.
  std::size_t slash_found = dirPath.rfind("/");
  if (slash_found != std::string::npos) {
    ftwBuf.base = slash_found + 1;
  }

  ftwBuf.level = level;

  // If the dirPath is a file, call the function on it and return.
  if ((typeFlag == FTW_SL) || (typeFlag == FTW_F)) {
    int returnValue = fn.call(dirPath.c_str(), &statBuf, typeFlag, &ftwBuf);
    if (returnValue) {
      return returnValue;
    }
    return 0;
  }

  // If FTW_DEPTH is not set, nftw() shall report any directory before reporting
  // the files in that directory.
  if (!(flags & FTW_DEPTH)) {
    // Call the function on the directory.
    int directory_fd = open(dirPath.c_str(), O_RDONLY);
    if (directory_fd < 0 && errno == EACCES) {
      typeFlag = FTW_DNR; /* Directory can't be read. */
    }
    close(directory_fd);

    int returnValue = fn.call(dirPath.c_str(), &statBuf, typeFlag, &ftwBuf);
    if (returnValue) {
      return returnValue;
    }
  }

  for (std::error_code ec; auto const& dir_entry :
                           std::filesystem::directory_iterator(dirPath, ec)) {
    if (ec) continue;
    int returnValue =
        doMergedFtw(dir_entry.path(), fn, fdLimit - 1, flags, ftwBuf.level + 1);
    if (returnValue) {
      return returnValue;
    }
  }

  // If FTW_DEPTH is set, nftw() shall report all files in a directory before
  // reporting the directory itself.
  if (flags & FTW_DEPTH) {
    // Call the function on the directory.
    return fn.call(dirPath.c_str(), &statBuf, typeFlag, &ftwBuf);
  }
  return 0;
}
} // namespace

int llvm_libc_nftw(const std::string& dirPath,
                   int (*fn)(const char* filePath, const struct stat* statBuf,
                             int tFlag, struct FTW* ftwbuf),
                   int fdLimit, int flags) {
  NftwFunc wrappedFn{fn};
  return doMergedFtw(dirPath, wrappedFn, fdLimit, flags, 0);
}

int llvm_libc_ftw(const std::string &dirPath,
                  int (*fn)(const char *filePath, const struct stat *statBuf,
                            int tFlag),
                  int fdLimit) {
  return llvm_libc_nftw(dirPath, (int (*)())fn, fdLimit, FTW_PHYS, 0);
}
