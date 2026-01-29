//===--- DarwinSDKInfo.h - SDK Information parser for darwin ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_DARWINSDKINFO_H
#define LLVM_CLANG_BASIC_DARWINSDKINFO_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/Triple.h"
#include <optional>
#include <string>

namespace llvm {
namespace json {
class Object;
} // end namespace json
} // end namespace llvm

namespace clang {

/// The information about the darwin SDK that was used during this compilation.
class DarwinSDKInfo {
public:
  /// Information about the supported platforms, derived from the target triple
  /// definitions, in the SDK.
  struct SDKPlatformInfo {
  public:
    SDKPlatformInfo(llvm::Triple::VendorType Vendor, llvm::Triple::OSType OS,
                    llvm::Triple::EnvironmentType Environment,
                    llvm::Triple::ObjectFormatType ObjectFormat,
                    StringRef PlatformPrefix)
        : Vendor(Vendor), OS(OS), Environment(Environment),
          ObjectFormat(ObjectFormat), PlatformPrefix(PlatformPrefix) {}

    llvm::Triple::VendorType getVendor() const { return Vendor; }
    llvm::Triple::OSType getOS() const { return OS; }
    llvm::Triple::EnvironmentType getEnvironment() const { return Environment; }
    llvm::Triple::ObjectFormatType getObjectFormat() const {
      return ObjectFormat;
    }
    StringRef getPlatformPrefix() const { return PlatformPrefix; }

    bool operator==(const llvm::Triple &RHS) const {
      return (Vendor == RHS.getVendor()) && (OS == RHS.getOS()) &&
             (Environment == RHS.getEnvironment()) &&
             (ObjectFormat == RHS.getObjectFormat());
    }

  private:
    llvm::Triple::VendorType Vendor;
    llvm::Triple::OSType OS;
    llvm::Triple::EnvironmentType Environment;
    llvm::Triple::ObjectFormatType ObjectFormat;
    std::string PlatformPrefix;
  };

  /// A value that describes two os-environment pairs that can be used as a key
  /// to the version map in the SDK.
  struct OSEnvPair {
  public:
    using StorageType = uint64_t;

    constexpr OSEnvPair(llvm::Triple::OSType FromOS,
                        llvm::Triple::EnvironmentType FromEnv,
                        llvm::Triple::OSType ToOS,
                        llvm::Triple::EnvironmentType ToEnv)
        : Value(((StorageType(FromOS) * StorageType(llvm::Triple::LastOSType) +
                  StorageType(FromEnv))
                 << 32ull) |
                (StorageType(ToOS) * StorageType(llvm::Triple::LastOSType) +
                 StorageType(ToEnv))) {}

    /// Returns the os-environment mapping pair that's used to represent the
    /// macOS -> Mac Catalyst version mapping.
    static inline constexpr OSEnvPair macOStoMacCatalystPair() {
      return OSEnvPair(llvm::Triple::MacOSX, llvm::Triple::UnknownEnvironment,
                       llvm::Triple::IOS, llvm::Triple::MacABI);
    }

    /// Returns the os-environment mapping pair that's used to represent the
    /// Mac Catalyst -> macOS version mapping.
    static inline constexpr OSEnvPair macCatalystToMacOSPair() {
      return OSEnvPair(llvm::Triple::IOS, llvm::Triple::MacABI,
                       llvm::Triple::MacOSX, llvm::Triple::UnknownEnvironment);
    }

    /// Returns the os-environment mapping pair that's used to represent the
    /// iOS -> watchOS version mapping.
    static inline constexpr OSEnvPair iOStoWatchOSPair() {
      return OSEnvPair(llvm::Triple::IOS, llvm::Triple::UnknownEnvironment,
                       llvm::Triple::WatchOS, llvm::Triple::UnknownEnvironment);
    }

    /// Returns the os-environment mapping pair that's used to represent the
    /// iOS -> tvOS version mapping.
    static inline constexpr OSEnvPair iOStoTvOSPair() {
      return OSEnvPair(llvm::Triple::IOS, llvm::Triple::UnknownEnvironment,
                       llvm::Triple::TvOS, llvm::Triple::UnknownEnvironment);
    }

  private:
    StorageType Value;

    friend class DarwinSDKInfo;
  };

  /// Represents a version mapping that maps from a version of one target to a
  /// version of a related target.
  ///
  /// e.g. "macOS_iOSMac":{"10.15":"13.1"} is an example of a macOS -> Mac
  /// Catalyst version map.
  class RelatedTargetVersionMapping {
  public:
    RelatedTargetVersionMapping(
        VersionTuple MinimumKeyVersion, VersionTuple MaximumKeyVersion,
        VersionTuple MinimumValue, VersionTuple MaximumValue,
        llvm::DenseMap<VersionTuple, VersionTuple> Mapping)
        : MinimumKeyVersion(MinimumKeyVersion),
          MaximumKeyVersion(MaximumKeyVersion), MinimumValue(MinimumValue),
          MaximumValue(MaximumValue), Mapping(Mapping) {
      assert(!this->Mapping.empty() && "unexpected empty mapping");
    }

    /// Returns the value with the lowest version in the mapping.
    const VersionTuple &getMinimumValue() const { return MinimumValue; }

    /// Returns the mapped key, or the appropriate Minimum / MaximumValue if
    /// they key is outside of the mapping bounds. If they key isn't mapped, but
    /// within the minimum and maximum bounds, std::nullopt is returned.
    std::optional<VersionTuple>
    map(const VersionTuple &Key, const VersionTuple &MinimumValue,
        std::optional<VersionTuple> MaximumValue) const;

    /// Remap the 'introduced' availability version.
    /// If None is returned, the 'unavailable' availability should be used
    /// instead.
    std::optional<VersionTuple>
    mapIntroducedAvailabilityVersion(const VersionTuple &Key) const {
      // API_TO_BE_DEPRECATED is 100000.
      if (Key.getMajor() == 100000)
        return VersionTuple(100000);
      // Use None for maximum to force unavailable behavior for
      return map(Key, MinimumValue, std::nullopt);
    }

    /// Remap the 'deprecated' and 'obsoleted' availability version.
    /// If None is returned for 'obsoleted', the 'unavailable' availability
    /// should be used instead. If None is returned for 'deprecated', the
    /// 'deprecated' version should be dropped.
    std::optional<VersionTuple>
    mapDeprecatedObsoletedAvailabilityVersion(const VersionTuple &Key) const {
      // API_TO_BE_DEPRECATED is 100000.
      if (Key.getMajor() == 100000)
        return VersionTuple(100000);
      return map(Key, MinimumValue, MaximumValue);
    }

    static std::optional<RelatedTargetVersionMapping>
    parseJSON(const llvm::json::Object &Obj,
              VersionTuple MaximumDeploymentTarget);

  private:
    VersionTuple MinimumKeyVersion;
    VersionTuple MaximumKeyVersion;
    VersionTuple MinimumValue;
    VersionTuple MaximumValue;
    llvm::DenseMap<VersionTuple, VersionTuple> Mapping;
  };

  using PlatformInfoStorageType = SmallVector<SDKPlatformInfo, 2>;

  DarwinSDKInfo(
      std::string FilePath, llvm::Triple::OSType OS,
      llvm::Triple::EnvironmentType Environment, VersionTuple Version,
      StringRef DisplayName, VersionTuple MaximumDeploymentTarget,
      PlatformInfoStorageType PlatformInfos,
      llvm::DenseMap<OSEnvPair::StorageType,
                     std::optional<RelatedTargetVersionMapping>>
          VersionMappings =
              llvm::DenseMap<OSEnvPair::StorageType,
                             std::optional<RelatedTargetVersionMapping>>())
      : FilePath(FilePath), OS(OS), Environment(Environment), Version(Version),
        DisplayName(DisplayName),
        MaximumDeploymentTarget(MaximumDeploymentTarget),
        PlatformInfos(std::move(PlatformInfos)),
        VersionMappings(std::move(VersionMappings)) {}

  StringRef getFilePath() const { return FilePath; }

  llvm::Triple::OSType getOS() const { return OS; }

  llvm::Triple::EnvironmentType getEnvironment() const { return Environment; }

  const llvm::VersionTuple &getVersion() const { return Version; }

  const StringRef getDisplayName() const { return DisplayName; }

  const SDKPlatformInfo &getCanonicalPlatformInfo() const {
    return PlatformInfos[0];
  }

  bool supportsTriple(llvm::Triple Triple) const {
    return llvm::find(PlatformInfos, Triple) != PlatformInfos.end();
  }

  const StringRef getPlatformPrefix(llvm::Triple Triple) const {
    auto PlatformInfoIt = llvm::find(PlatformInfos, Triple);
    if (PlatformInfoIt == PlatformInfos.end())
      return StringRef();
    return PlatformInfoIt->getPlatformPrefix();
  }

  // Returns the optional, target-specific version mapping that maps from one
  // target to another target.
  //
  // This mapping is constructed from an appropriate mapping in the SDKSettings,
  // for instance, when building for Mac Catalyst, the mapping would contain the
  // "macOS_iOSMac" mapping as it maps the macOS versions to the Mac Catalyst
  // versions.
  //
  // This mapping does not exist when the target doesn't have an appropriate
  // related version mapping, or when there was an error reading the mapping
  // from the SDKSettings, or when it's missing in the SDKSettings.
  const RelatedTargetVersionMapping *getVersionMapping(OSEnvPair Kind) const {
    auto Mapping = VersionMappings.find(Kind.Value);
    if (Mapping == VersionMappings.end())
      return nullptr;
    return Mapping->getSecond() ? &*Mapping->getSecond() : nullptr;
  }

  static std::optional<DarwinSDKInfo>
  parseDarwinSDKSettingsJSON(std::string FilePath,
                             const llvm::json::Object *Obj);

private:
  std::string FilePath;
  llvm::Triple::OSType OS;
  llvm::Triple::EnvironmentType Environment;
  VersionTuple Version;
  std::string DisplayName;
  VersionTuple MaximumDeploymentTarget;
  PlatformInfoStorageType PlatformInfos;
  // Need to wrap the value in an optional here as the value has to be default
  // constructible, and std::unique_ptr doesn't like DarwinSDKInfo being
  // Optional as Optional is trying to copy it in emplace.
  llvm::DenseMap<OSEnvPair::StorageType,
                 std::optional<RelatedTargetVersionMapping>>
      VersionMappings;
};

/// Parse the SDK information from the SDKSettings.json file.
///
/// \returns an error if the SDKSettings.json file is invalid, std::nullopt if
/// the SDK has no SDKSettings.json, or a valid \c DarwinSDKInfo otherwise.
Expected<std::optional<DarwinSDKInfo>>
parseDarwinSDKInfo(llvm::vfs::FileSystem &VFS, StringRef SDKRootPath);

} // end namespace clang

#endif // LLVM_CLANG_BASIC_DARWINSDKINFO_H
