//===-- Implementation of shared ftw/nftw logic ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ftw/ftw_impl.h"

#include "src/__support/CPP/string.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/File/scoped_dir.h"
#include "src/__support/libc_errno.h"
#include "src/fcntl/open.h"
#include "src/sys/stat/lstat.h"
#include "src/sys/stat/stat.h"
#include "src/unistd/chdir.h"
#include "src/unistd/close.h"
#include "src/unistd/fchdir.h"

#include "hdr/fcntl_macros.h"
#include "hdr/ftw_macros.h"
#include "hdr/sys_stat_macros.h"
#ifdef LIBC_FULL_BUILD
#include "include/llvm-libc-types/struct_FTW.h"
#include "include/llvm-libc-types/struct_dirent.h"
#include "include/llvm-libc-types/struct_stat.h"
#else
#include <dirent.h>
#include <ftw.h>
#include <sys/stat.h>
#endif

namespace LIBC_NAMESPACE_DECL {
namespace ftw_impl {

class StartDirSaver {
  int StartFd;

public:
  StartDirSaver(int Fd) : StartFd(Fd) {}
  ~StartDirSaver() {
    if (StartFd >= 0) {
      fchdir(StartFd);
      close(StartFd);
    }
  }
};

class LevelDirSaver {
  bool Active;

public:
  LevelDirSaver(bool DoChdir) : Active(DoChdir) {}
  ~LevelDirSaver() {
    if (Active)
      chdir("..");
  }
};

cpp::expected<int, int> doMergedFtw(const cpp::string &DirPath,
                                    const CallbackWrapper &Fn, int FdLimit,
                                    int Flags, int Level,
                                    unsigned long StartDevice,
                                    AncestorDir *Ancestors) {
  int StartFd = -1;
  // Save starting directory for FTW_CHDIR restoration.
  if (Level == 0 && (Flags & FTW_CHDIR)) {
    StartFd = open(".", O_RDONLY);
    if (StartFd < 0)
      return cpp::unexpected<int>(libc_errno);
  }
  StartDirSaver RootSaver(StartFd);

  // Set up FTW buffer and calculate filename offset base.
  struct FTW FtwBuf;
  FtwBuf.level = Level;
  cpp::string_view PathView(DirPath);
  size_t SlashFound = PathView.find_last_of('/');
  FtwBuf.base = (SlashFound != cpp::string_view::npos)
                    ? static_cast<int>(SlashFound + 1)
                    : 0;

  const char *OsPath = (Level == 0 || !(Flags & FTW_CHDIR))
                           ? DirPath.c_str()
                           : (DirPath.c_str() + FtwBuf.base);

  int TypeFlag = FTW_F;
  struct stat StatBuf;
  // Stat the path, respecting FTW_PHYS with lstat vs stat.
  // We use LIBC_NAMESPACE:: so that we call the internal (l)stat in overlay
  // mode.
  if (Flags & FTW_PHYS) {
    if (LIBC_NAMESPACE::lstat(OsPath, &StatBuf) < 0) {
      if (libc_errno == EACCES)
        TypeFlag = FTW_NS;
      else
        return cpp::unexpected<int>(libc_errno);
    }
  } else {
    if (LIBC_NAMESPACE::stat(OsPath, &StatBuf) < 0) {
      if (libc_errno == EACCES) {
        TypeFlag = FTW_NS;
      } else if (LIBC_NAMESPACE::lstat(OsPath, &StatBuf) == 0) {
        // Dangling symlink found.
        TypeFlag = FTW_SLN;
      } else if (libc_errno == EACCES) {
        TypeFlag = FTW_NS;
      } else {
        return cpp::unexpected<int>(libc_errno);
      }
    }
  }

  // Track starting device for FTW_MOUNT traversal limits.
  if (Level == 0)
    StartDevice = StatBuf.st_dev;

  // Skip traversal into mounted filesystems if FTW_MOUNT is set.
  if ((Flags & FTW_MOUNT) && Level > 0 && StatBuf.st_dev != StartDevice)
    return 0;

  // Map stat mode to final FTW_* type flags.
  if (TypeFlag != FTW_SLN && TypeFlag != FTW_NS) {
    if (S_ISDIR(StatBuf.st_mode))
      TypeFlag = (Flags & FTW_DEPTH) ? FTW_DP : FTW_D;
    else if (S_ISLNK(StatBuf.st_mode))
      TypeFlag = (Flags & FTW_PHYS) ? FTW_SL : FTW_SLN;
    else
      TypeFlag = FTW_F;
  }

  // Legacy ftw() must map FTW_SLN to FTW_SL.
  if (!Fn.IsNftw && TypeFlag == FTW_SLN)
    TypeFlag = FTW_SL;

  // Cycle detection for directories to prevent infinite recursion.
  if (TypeFlag == FTW_D || TypeFlag == FTW_DP || TypeFlag == FTW_DNR) {
    for (AncestorDir *A = Ancestors; A != nullptr; A = A->Parent) {
      if (A->Dev == StatBuf.st_dev && A->Ino == StatBuf.st_ino)
        return 0;
    }
  }

  bool SkipSubtree = false;
  Dir *OpenDir = nullptr;
  // Attempt directory open; propagate fd count exhaustion errors.
  if (TypeFlag == FTW_D || TypeFlag == FTW_DP) {
    if (FdLimit <= 0)
      return cpp::unexpected<int>(EMFILE);
    auto DirResult = Dir::open(OsPath);
    if (!DirResult) {
      if (DirResult.error() == EACCES) {
        TypeFlag = FTW_DNR;
      } else {
        return cpp::unexpected<int>(DirResult.error());
      }
    } else {
      OpenDir = DirResult.value();
    }
  }

  if (TypeFlag != FTW_D && TypeFlag != FTW_DP)
    return Fn.call(DirPath.c_str(), &StatBuf, TypeFlag, &FtwBuf);

  // Pre-order traversal: call callback BEFORE descending.
  if (!(Flags & FTW_DEPTH)) {
    int Ret = Fn.call(DirPath.c_str(), &StatBuf, TypeFlag, &FtwBuf);
    if (Ret != 0) {
      if (Flags & FTW_ACTIONRETVAL) {
        // Honor action return values if requested.
        if (Ret == FTW_SKIP_SUBTREE) {
          SkipSubtree = true;
        } else if (Ret == FTW_SKIP_SIBLINGS) {
          if (OpenDir)
            OpenDir->close(); // ScopedDir not yet created
          return Ret;
        } else {
          if (OpenDir)
            OpenDir->close();
          return Ret;
        }
      } else {
        if (OpenDir)
          OpenDir->close();
        return Ret;
      }
    }
  }

  // Descend into children.
  if (OpenDir && !SkipSubtree) {
    ScopedDir DirGuard(OpenDir);
    if (Flags & FTW_CHDIR) {
      if (chdir(OsPath) < 0)
        return cpp::unexpected<int>(libc_errno);
    }
    LevelDirSaver LevelSaver(Flags & FTW_CHDIR);
    AncestorDir CurrentAncestor = {StatBuf.st_dev, StatBuf.st_ino, Ancestors};

    while (true) {
      auto Entry = DirGuard->read();
      if (!Entry)
        return cpp::unexpected(Entry.error());

      struct ::dirent *DirentPtr = Entry.value();
      if (DirentPtr == nullptr)
        break;

      // Skip dot and dot-dot directories.
      if (DirentPtr->d_name[0] == '.') {
        if (DirentPtr->d_name[1] == '\0' ||
            (DirentPtr->d_name[1] == '.' && DirentPtr->d_name[2] == '\0'))
          continue;
      }

      cpp::string EntryPath = DirPath;
      if (!EntryPath.empty() && EntryPath[EntryPath.size() - 1] != '/')
        EntryPath += "/";
      EntryPath += DirentPtr->d_name;

      auto Res = doMergedFtw(EntryPath, Fn, FdLimit - 1, Flags, Level + 1,
                             StartDevice, &CurrentAncestor);
      if (!Res)
        return Res;
      if (Flags & FTW_ACTIONRETVAL) {
        if (Res.value() == FTW_SKIP_SIBLINGS)
          break;
        if (Res.value() != 0 && Res.value() != FTW_SKIP_SUBTREE)
          return Res.value();
      } else if (Res.value() != 0) {
        return Res.value();
      }
    }
  } else if (OpenDir) {
    OpenDir->close();
  }

  // Post-order traversal: call callback AFTER descending.
  if ((Flags & FTW_DEPTH) && !SkipSubtree) {
    int Ret = Fn.call(DirPath.c_str(), &StatBuf, TypeFlag, &FtwBuf);
    if (Flags & FTW_ACTIONRETVAL) {
      if (Ret == FTW_SKIP_SIBLINGS || Ret == FTW_SKIP_SUBTREE)
        return Ret;
    }
    return Ret;
  }
  return 0;
}

} // namespace ftw_impl
} // namespace LIBC_NAMESPACE_DECL
