//===-- HostInfoMacOSX.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_MACOSX_HOSTINFOMACOSX_H
#define LLDB_HOST_MACOSX_HOSTINFOMACOSX_H

#include "lldb/Host/posix/HostInfoPosix.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/XcodeSDK.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VersionTuple.h"
#include <optional>

namespace lldb_private {

class ArchSpec;

class HostInfoMacOSX : public HostInfoPosix {
  friend class HostInfoBase;

public:
  static llvm::VersionTuple GetOSVersion();
  static llvm::VersionTuple GetMacCatalystVersion();
  static std::optional<std::string> GetOSBuildString();
  static FileSpec GetProgramFileSpec();
  static FileSpec GetXcodeContentsDirectory();
  static FileSpec GetXcodeDeveloperDirectory();
  static FileSpec GetCurrentXcodeToolchainDirectory();
  static FileSpec GetCurrentCommandLineToolsDirectory();

#ifdef LLDB_ENABLE_SWIFT
  static FileSpec GetSwiftResourceDir();
  static std::string GetSwiftResourceDir(llvm::Triple triple,
                                         llvm::StringRef platform_sdk_path);

  /// Return the name of the OS-specific subdirectory containing the
  /// Swift stdlib needed for \p target. Only exposed for unit tests.
  static std::string GetSwiftStdlibOSDir(llvm::Triple target,
                                         llvm::Triple host);

  /// Private implementation detail of GetSwiftResourceDir, exposed for unit
  /// tests.
  static std::string DetectSwiftResourceDir(llvm::StringRef platform_sdk_path,
                                            llvm::StringRef swift_stdlib_os_dir,
                                            std::string swift_dir,
                                            std::string xcode_contents_path,
                                            std::string toolchain_path,
                                            std::string cl_tools_path);
  static bool ComputeSwiftResourceDirectory(FileSpec &lldb_shlib_spec,
                                            FileSpec &file_spec, bool verify);
#endif

  /// Query xcrun to find an Xcode SDK directory.
  ///
  /// Note, this is an expensive operation if the SDK we're querying
  /// does not exist in an Xcode installation path on the host.
  static llvm::Expected<llvm::StringRef> GetSDKRoot(SDKOptions options);
  static llvm::Expected<llvm::StringRef> FindSDKTool(XcodeSDK sdk,
                                                     llvm::StringRef tool);

  /// Shared cache utilities
  static SharedCacheImageInfo
  GetSharedCacheImageInfo(llvm::StringRef image_name);

protected:
  static bool ComputeSupportExeDirectory(FileSpec &file_spec);
  static void ComputeHostArchitectureSupport(ArchSpec &arch_32,
                                             ArchSpec &arch_64);
  static bool ComputeHeaderDirectory(FileSpec &file_spec);
  static bool ComputeSystemPluginsDirectory(FileSpec &file_spec);
  static bool ComputeUserPluginsDirectory(FileSpec &file_spec);

  static std::string FindComponentInPath(llvm::StringRef path,
                                         llvm::StringRef component);
};
}

#endif
