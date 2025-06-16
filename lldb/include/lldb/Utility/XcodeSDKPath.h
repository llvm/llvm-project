//===-- XcodeSDK.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_SDK_PATH_H
#define LLDB_UTILITY_SDK_PATH_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/XcodeSDK.h"

namespace llvm {
class Triple;
}

namespace lldb_private {

/// An abstraction which groups an XcodeSDK with its parsed path.
class XcodeSDKPath {
  XcodeSDK m_sdk;
  FileSpec m_sysroot;

public:
  /// Default constructor, constructs an empty sdk with an empty path.
  XcodeSDKPath() = default;
  XcodeSDKPath(XcodeSDK sdk, FileSpec sysroot)
      : m_sdk(std::move(sdk)), m_sysroot(std::move(sysroot)) {}
  XcodeSDKPath(std::string name, FileSpec sysroot)
      : m_sdk(XcodeSDK(std::move(name))), m_sysroot(std::move(sysroot)) {}

  bool operator==(const XcodeSDKPath &other) const;
  bool operator!=(const XcodeSDKPath &other) const;

  XcodeSDK TakeSDK() const;
  const FileSpec &GetSysroot() const { return m_sysroot; }
  llvm::StringRef GetString() const { return m_sdk.GetString(); }
  XcodeSDK::Type GetType() const { return m_sdk.GetType(); }

  void Merge(const XcodeSDKPath &other);
  bool IsAppleInternalSDK() const { return m_sdk.IsAppleInternalSDK(); }
};

} // namespace lldb_private

#endif
