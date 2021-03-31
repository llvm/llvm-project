//===-- HostInfoBase.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_HOSTINFOBASE_H
#define LLDB_HOST_HOSTINFOBASE_H

#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/UUID.h"
#include "lldb/Utility/UserIDResolver.h"
#include "lldb/Utility/XcodeSDK.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringRef.h"

#include <stdint.h>

#include <string>

namespace lldb_private {

class FileSpec;

struct SharedCacheImageInfo {
  UUID uuid;
  lldb::DataBufferSP data_sp;
};

class HostInfoBase {
private:
  // Static class, unconstructable.
  HostInfoBase() {}
  ~HostInfoBase() {}

public:
  /// A helper function for determining the liblldb location. It receives a
  /// FileSpec with the location of file containing _this_ code. It can
  /// (optionally) replace it with a file spec pointing to a more canonical
  /// copy.
  using SharedLibraryDirectoryHelper = void(FileSpec &this_file);

  static void Initialize(SharedLibraryDirectoryHelper *helper = nullptr);
  static void Terminate();

  /// Gets the host target triple.
  ///
  /// \return
  ///     The host target triple.
  static llvm::Triple GetTargetTriple();

  enum ArchitectureKind {
    eArchKindDefault, // The overall default architecture that applications will
                      // run on this host
    eArchKind32, // If this host supports 32 bit programs, return the default 32
                 // bit arch
    eArchKind64  // If this host supports 64 bit programs, return the default 64
                 // bit arch
  };

  static const ArchSpec &
  GetArchitecture(ArchitectureKind arch_kind = eArchKindDefault);

  static llvm::Optional<ArchitectureKind> ParseArchitectureKind(llvm::StringRef kind);

  /// Returns the directory containing the lldb shared library. Only the
  /// directory member of the FileSpec is filled in.
  static FileSpec GetShlibDir();

  /// Returns the directory containing the support executables (debugserver,
  /// ...). Only the directory member of the FileSpec is filled in.
  static FileSpec GetSupportExeDir();

  /// Returns the directory containing the lldb headers. Only the directory
  /// member of the FileSpec is filled in.
  static FileSpec GetHeaderDir();

  /// Returns the directory containing the system plugins. Only the directory
  /// member of the FileSpec is filled in.
  static FileSpec GetSystemPluginDir();

  /// Returns the directory containing the user plugins. Only the directory
  /// member of the FileSpec is filled in.
  static FileSpec GetUserPluginDir();

  /// Returns the proces temporary directory. This directory will be cleaned up
  /// when this process exits. Only the directory member of the FileSpec is
  /// filled in.
  static FileSpec GetProcessTempDir();

  /// Returns the global temporary directory. This directory will **not** be
  /// cleaned up when this process exits. Only the directory member of the
  /// FileSpec is filled in.
  static FileSpec GetGlobalTempDir();

  /// If the triple does not specify the vendor, os, and environment parts, we
  /// "augment" these using information from the host and return the resulting
  /// ArchSpec object.
  static ArchSpec GetAugmentedArchSpec(llvm::StringRef triple);

  static bool ComputePathRelativeToLibrary(FileSpec &file_spec,
                                           llvm::StringRef dir);

  static FileSpec GetXcodeContentsDirectory() { return {}; }
  static FileSpec GetXcodeDeveloperDirectory() { return {}; }
  
  /// Return the directory containing a specific Xcode SDK.
  static llvm::StringRef GetXcodeSDKPath(XcodeSDK sdk) { return {}; }

  /// Return information about module \p image_name if it is loaded in
  /// the current process's address space.
  static SharedCacheImageInfo
  GetSharedCacheImageInfo(llvm::StringRef image_name) {
    return {};
  }

protected:
  static bool ComputeSharedLibraryDirectory(FileSpec &file_spec);
  static bool ComputeSupportExeDirectory(FileSpec &file_spec);
  static bool ComputeProcessTempFileDirectory(FileSpec &file_spec);
  static bool ComputeGlobalTempFileDirectory(FileSpec &file_spec);
  static bool ComputeTempFileBaseDirectory(FileSpec &file_spec);
  static bool ComputeHeaderDirectory(FileSpec &file_spec);
  static bool ComputeSystemPluginsDirectory(FileSpec &file_spec);
  static bool ComputeUserPluginsDirectory(FileSpec &file_spec);

  static void ComputeHostArchitectureSupport(ArchSpec &arch_32,
                                             ArchSpec &arch_64);
};
}

#endif
