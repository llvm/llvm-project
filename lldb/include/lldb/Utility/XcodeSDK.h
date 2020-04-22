//===-- XcodeSDK.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_SDK_H
#define LLDB_UTILITY_SDK_H

#include "lldb/lldb-forward.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VersionTuple.h"
#include <tuple>

namespace lldb_private {

/// An abstraction for Xcode-style SDKs that works like \ref ArchSpec.
class XcodeSDK {
  std::string m_name;

public:
  XcodeSDK() = default;
  XcodeSDK(std::string &&name) : m_name(std::move(name)) {}
  static XcodeSDK GetAnyMacOS() { return XcodeSDK("MacOSX.sdk"); }

  enum Type : int {
    MacOSX = 0,
    iPhoneSimulator,
    iPhoneOS,
    AppleTVSimulator,
    AppleTVOS,
    WatchSimulator,
    watchOS,
    bridgeOS,
    Linux,
    numSDKTypes,
    unknown = -1
  };
  static llvm::StringRef GetNameForType(Type type);

  /// The merge function follows a strict order to maintain monotonicity:
  /// 1. SDK with the higher SDKType wins.
  /// 2. The newer SDK wins.
  void Merge(XcodeSDK other);

  XcodeSDK &operator=(XcodeSDK other);
  XcodeSDK(const XcodeSDK&) = default;
  bool operator==(XcodeSDK other);

  /// Return parsed SDK number, and SDK version number.
  std::tuple<Type, llvm::VersionTuple> Parse() const;
  llvm::VersionTuple GetVersion() const;
  Type GetType() const;
  llvm::StringRef GetString() const;

  static bool SDKSupportsModules(Type type, llvm::VersionTuple version);
  static bool SDKSupportsModules(Type desired_type, const FileSpec &sdk_path);
  static llvm::StringRef GetSDKNameForType(Type type);
};

} // namespace lldb_private

#endif
