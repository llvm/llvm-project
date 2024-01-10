/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "comgr-env.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <fstream>
#include <stdlib.h>

using namespace llvm;

namespace COMGR {
namespace env {

bool shouldSaveTemps() {
  static char *SaveTemps = getenv("AMD_COMGR_SAVE_TEMPS");
  return SaveTemps && StringRef(SaveTemps) != "0";
}

std::optional<StringRef> getRedirectLogs() {
  static char *RedirectLogs = getenv("AMD_COMGR_REDIRECT_LOGS");
  if (!RedirectLogs || StringRef(RedirectLogs) == "0") {
    return std::nullopt;
  }
  return StringRef(RedirectLogs);
}

bool needTimeStatistics() {
  static char *TimeStatistics = getenv("AMD_COMGR_TIME_STATISTICS");
  return TimeStatistics && StringRef(TimeStatistics) != "0";
}

bool shouldEmitVerboseLogs() {
  static char *VerboseLogs = getenv("AMD_COMGR_EMIT_VERBOSE_LOGS");
  return VerboseLogs && StringRef(VerboseLogs) != "0";
}

StringRef StripGNUInstallLibDir(StringRef Path) {
  // Comgr library may be installed under lib or lib64 or
  // lib/<multiarch-tuple> on Debian.
  StringRef ParentDir = llvm::sys::path::parent_path(Path);
  StringRef ParentName = llvm::sys::path::filename(ParentDir);

  StringRef SecondLevelParentDir = llvm::sys::path::parent_path(ParentDir);
  StringRef SecondLevelParentName =
      llvm::sys::path::filename(SecondLevelParentDir);

  if (ParentName == "lib") {
    ParentDir = llvm::sys::path::parent_path(ParentDir);
    ParentName = llvm::sys::path::filename(ParentDir);
  } else if (ParentName == "lib64") {
    ParentDir = llvm::sys::path::parent_path(ParentDir);
    ParentName = llvm::sys::path::filename(ParentDir);
  } else if (SecondLevelParentName == "lib") {
    ParentDir = SecondLevelParentDir;
  }

  return ParentDir;
}

std::string getComgrInstallPathFromExecutable() {

#if !defined(_WIN32) && !defined(_WIN64)
  FILE *ProcMaps = fopen("/proc/self/maps", "r");
  if (ProcMaps == NULL)
    return "";

  char *Line = NULL;
  size_t len = 0;
  uintptr_t Address = reinterpret_cast<uintptr_t>(getROCMPath);

  // TODO: switch POSIX getline() to C++-based getline() once Pytorch resolves
  // build issues with libstdc++ ABI
  while (getline(&Line, &len, ProcMaps) != -1) {
    llvm::SmallVector<StringRef, 6> Tokens;
    StringRef(Line).split(Tokens, ' ', -1 /* MaxSplit */,
                          false /* KeepEmpty */);

    unsigned long long LowAddress, HighAddress;
    if (llvm::consumeUnsignedInteger(Tokens[0], 16 /* Radix */, LowAddress)) {
      fclose(ProcMaps);
      free(Line);
      return "";
    }

    if (!Tokens[0].consume_front("-")) {
      fclose(ProcMaps);
      free(Line);
      return "";
    }

    if (llvm::consumeUnsignedInteger(Tokens[0], 16 /* Radix */, HighAddress)) {
      fclose(ProcMaps);
      free(Line);
      return "";
    }

    if ((Address >= LowAddress && Address <= HighAddress)) {
      StringRef Path = Tokens[5].ltrim();
      /* Not a mapped file or File path empty */
      if (Tokens[4] == "0" || Path == "") {
        fclose(ProcMaps);
        free(Line);
        return "";
      }

      std::string rv = StripGNUInstallLibDir(Path).str();
      fclose(ProcMaps);
      free(Line);
      return rv;
    }
  }

  fclose(ProcMaps);
  free(Line);
#endif

  return "";
}

class InstallationDetector {
public:
  InstallationDetector(StringRef ROCmPath, bool isComgrPath)
      : ROCmInstallPath(ROCmPath) {}
  virtual ~InstallationDetector() = default;

  const StringRef getROCmPath() const { return ROCmInstallPath; }
  void setROCmInstallPath(StringRef Path) { ROCmInstallPath = Path; }

  virtual SmallString<128> getLLVMPathImpl() {
    SmallString<128> LLVMPath = getROCmPath();
    sys::path::append(LLVMPath, "llvm");

    return LLVMPath;
  }

  virtual SmallString<128> getHIPPathImpl() {
    SmallString<128> HIPPath = getROCmPath();
    sys::path::append(HIPPath, "hip");

    return HIPPath;
  }

  StringRef getLLVMPath() {
    static const char *EnvLLVMPath = std::getenv("LLVM_PATH");
    if (EnvLLVMPath) {
      return EnvLLVMPath;
    }

    if (LLVMInstallationPath.empty()) {
      LLVMInstallationPath = getLLVMPathImpl();
    }

    return LLVMInstallationPath;
  }

  StringRef getHIPPath() {
    static const char *EnvHIPPath = std::getenv("HIP_PATH");
    if (EnvHIPPath) {
      return EnvHIPPath;
    }

    if (HIPInstallationPath.empty()) {
      HIPInstallationPath = getHIPPathImpl();
    }

    return HIPInstallationPath;
  }

  SmallString<128> getSiblingDirWithPrefix(StringRef DirName,
                                           StringRef Prefix) {
    StringRef ParentDir = llvm::sys::path::parent_path(DirName);
    std::error_code EC;

    for (sys::fs::directory_iterator Dir(ParentDir, EC), DirEnd;
         Dir != DirEnd && !EC; Dir.increment(EC)) {
      const StringRef Path = llvm::sys::path::filename(Dir->path());
      if (Path.starts_with(Prefix)) {
        return StringRef(Dir->path());
      }
    }

    return SmallString<128>();
  }

private:
  SmallString<128> ROCmInstallPath;
  SmallString<128> HIPInstallationPath;
  SmallString<128> LLVMInstallationPath;
};

// If the ROCmInstallPath is Spack based it should be in the format
// rocm-cmake-${rocm-version}-${hash}. Detect corresponding LLVM and HIP
// Paths existing at the same level at ROCM. It should be in the format
// llvm-amdgpu-${rocm-version}-${hash} and hip-${rocm-version}-${hash}.
class SpackInstallationDetector : public InstallationDetector {
public:
  SpackInstallationDetector(StringRef Path, bool isComgrPath)
      : InstallationDetector(Path, isComgrPath) {
    if (isComgrPath) {
      auto ROCmInstallPath = getSiblingDirWithPrefix(Path, "rocm-cmake-");
      setROCmInstallPath(ROCmInstallPath);
    }
  }

  virtual SmallString<128> getLLVMPathImpl() override {
    return getSiblingDirWithPrefix(getROCmPath(), "llvm-amdgpu-");
  }

  virtual SmallString<128> getHIPPathImpl() override {
    return getSiblingDirWithPrefix(getROCmPath(), "hip-");
  }
};

InstallationDetector *CreatePathDetector(StringRef Path,
                                         bool isComgrPath = false) {
  StringRef DirName = llvm::sys::path::filename(Path);
  if ((!isComgrPath && DirName.starts_with("rocm-cmake-")) ||
      (isComgrPath && DirName.starts_with("comgr-"))) {
    return new SpackInstallationDetector(Path, isComgrPath);
  }

  return new InstallationDetector(Path, isComgrPath);
}

InstallationDetector *getDetectorImpl() {
  SmallString<128> ROCmInstallPath;

  static const char *EnvROCMPath = std::getenv("ROCM_PATH");
  if (EnvROCMPath) {
    ROCmInstallPath = EnvROCMPath;
  }

  InstallationDetector *Detector;
  if (ROCmInstallPath == "") {
    std::string ComgrInstallationPath = getComgrInstallPathFromExecutable();
    Detector =
        CreatePathDetector(ComgrInstallationPath, true /* isComgrPath */);
  } else {
    Detector = CreatePathDetector(ROCmInstallPath);
  }

  return Detector;
}

InstallationDetector *getDetector() {
  static InstallationDetector *Detector = getDetectorImpl();
  return Detector;
}

llvm::StringRef getROCMPath() { return getDetector()->getROCmPath(); }

llvm::StringRef getHIPPath() { return getDetector()->getHIPPath(); }

llvm::StringRef getLLVMPath() { return getDetector()->getLLVMPath(); }

} // namespace env
} // namespace COMGR
