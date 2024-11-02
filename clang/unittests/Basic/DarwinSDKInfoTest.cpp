//===- unittests/Basic/DarwinSDKInfoTest.cpp -- SDKSettings.json test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DarwinSDKInfo.h"
#include "llvm/Support/JSON.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

// Check the version mapping logic in DarwinSDKInfo.
TEST(DarwinSDKInfo, VersionMapping) {
  llvm::json::Object Obj({{"3.0", "1.0"}, {"3.1", "1.2"}});
  Optional<DarwinSDKInfo::RelatedTargetVersionMapping> Mapping =
      DarwinSDKInfo::RelatedTargetVersionMapping::parseJSON(Obj,
                                                            VersionTuple());
  EXPECT_TRUE(Mapping);
  EXPECT_EQ(Mapping->getMinimumValue(), VersionTuple(1));

  // Exact mapping.
  EXPECT_EQ(Mapping->map(VersionTuple(3), VersionTuple(0, 1), std::nullopt),
            VersionTuple(1));
  EXPECT_EQ(Mapping->map(VersionTuple(3, 0), VersionTuple(0, 1), std::nullopt),
            VersionTuple(1));
  EXPECT_EQ(
      Mapping->map(VersionTuple(3, 0, 0), VersionTuple(0, 1), std::nullopt),
      VersionTuple(1));
  EXPECT_EQ(Mapping->map(VersionTuple(3, 1), VersionTuple(0, 1), std::nullopt),
            VersionTuple(1, 2));
  EXPECT_EQ(
      Mapping->map(VersionTuple(3, 1, 0), VersionTuple(0, 1), std::nullopt),
      VersionTuple(1, 2));

  // Missing mapping - fallback to major.
  EXPECT_EQ(
      Mapping->map(VersionTuple(3, 0, 1), VersionTuple(0, 1), std::nullopt),
      VersionTuple(1));

  // Minimum
  EXPECT_EQ(Mapping->map(VersionTuple(2), VersionTuple(0, 1), std::nullopt),
            VersionTuple(0, 1));

  // Maximum
  EXPECT_EQ(
      Mapping->map(VersionTuple(4), VersionTuple(0, 1), VersionTuple(100)),
      VersionTuple(100));
}

// Check the version mapping logic in DarwinSDKInfo.
TEST(DarwinSDKInfo, VersionMappingMissingKey) {
  llvm::json::Object Obj({{"3.0", "1.0"}, {"5.0", "1.2"}});
  Optional<DarwinSDKInfo::RelatedTargetVersionMapping> Mapping =
      DarwinSDKInfo::RelatedTargetVersionMapping::parseJSON(Obj,
                                                            VersionTuple());
  EXPECT_TRUE(Mapping);
  EXPECT_EQ(
      Mapping->map(VersionTuple(4), VersionTuple(0, 1), VersionTuple(100)),
      std::nullopt);
}

TEST(DarwinSDKInfo, VersionMappingParseEmpty) {
  llvm::json::Object Obj({});
  EXPECT_FALSE(
      DarwinSDKInfo::RelatedTargetVersionMapping::parseJSON(Obj, VersionTuple())
          .has_value());
}

TEST(DarwinSDKInfo, VersionMappingParseError) {
  llvm::json::Object Obj({{"test", "1.2"}});
  EXPECT_FALSE(
      DarwinSDKInfo::RelatedTargetVersionMapping::parseJSON(Obj, VersionTuple())
          .has_value());
}

TEST(DarwinSDKInfoTest, ParseAndTestMappingMacCatalyst) {
  llvm::json::Object Obj;
  Obj["Version"] = "11.0";
  Obj["MaximumDeploymentTarget"] = "11.99";
  llvm::json::Object VersionMap;
  VersionMap["10.15"] = "13.1";
  VersionMap["11.0"] = "14.0";
  VersionMap["11.2"] = "14.2";
  llvm::json::Object MacOS2iOSMac;
  MacOS2iOSMac["macOS_iOSMac"] = std::move(VersionMap);
  Obj["VersionMap"] = std::move(MacOS2iOSMac);

  auto SDKInfo = DarwinSDKInfo::parseDarwinSDKSettingsJSON(&Obj);
  ASSERT_TRUE(SDKInfo);
  EXPECT_EQ(SDKInfo->getVersion(), VersionTuple(11, 0));

  auto Mapping = SDKInfo->getVersionMapping(
      DarwinSDKInfo::OSEnvPair::macOStoMacCatalystPair());
  ASSERT_TRUE(Mapping);
  // Verify that the macOS versions that are present in the map are translated
  // directly to their corresponding Mac Catalyst versions.
  EXPECT_EQ(*Mapping->map(VersionTuple(10, 15), VersionTuple(), std::nullopt),
            VersionTuple(13, 1));
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 0), VersionTuple(), std::nullopt),
            VersionTuple(14, 0));
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 2), VersionTuple(), std::nullopt),
            VersionTuple(14, 2));

  // Verify that a macOS version that's not present in the map is translated
  // like the nearest major OS version.
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 1), VersionTuple(), std::nullopt),
            VersionTuple(14, 0));

  // Verify that the macOS versions that are outside of the mapped version
  // range map to the min/max values passed to the `map` call.
  EXPECT_EQ(
      *Mapping->map(VersionTuple(10, 14), VersionTuple(99, 99), std::nullopt),
      VersionTuple(99, 99));
  EXPECT_EQ(
      *Mapping->map(VersionTuple(11, 5), VersionTuple(), VersionTuple(99, 99)),
      VersionTuple(99, 99));
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 5), VersionTuple(99, 98),
                          VersionTuple(99, 99)),
            VersionTuple(99, 99));
}

TEST(DarwinSDKInfoTest, ParseAndTestMappingIOSDerived) {
  llvm::json::Object Obj;
  Obj["Version"] = "15.0";
  Obj["MaximumDeploymentTarget"] = "15.0.99";
  llvm::json::Object VersionMap;
  VersionMap["10.0"] = "10.0";
  VersionMap["10.3.1"] = "10.2";
  VersionMap["11.0"] = "11.0";
  llvm::json::Object IOSToTvOS;
  IOSToTvOS["iOS_tvOS"] = std::move(VersionMap);
  Obj["VersionMap"] = std::move(IOSToTvOS);

  auto SDKInfo = DarwinSDKInfo::parseDarwinSDKSettingsJSON(&Obj);
  ASSERT_TRUE(SDKInfo);
  EXPECT_EQ(SDKInfo->getVersion(), VersionTuple(15, 0));

  // Verify that mapping is present for platforms that derive from iOS.
  const auto *Mapping = SDKInfo->getVersionMapping(DarwinSDKInfo::OSEnvPair(
      llvm::Triple::IOS, llvm::Triple::UnknownEnvironment, llvm::Triple::TvOS,
      llvm::Triple::UnknownEnvironment));
  ASSERT_TRUE(Mapping);

  // Verify that the iOS versions that are present in the map are translated
  // directly to their corresponding tvOS versions.
  EXPECT_EQ(*Mapping->map(VersionTuple(10, 0), VersionTuple(), std::nullopt),
            VersionTuple(10, 0));
  EXPECT_EQ(*Mapping->map(VersionTuple(10, 3, 1), VersionTuple(), std::nullopt),
            VersionTuple(10, 2));
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 0), VersionTuple(), std::nullopt),
            VersionTuple(11, 0));

  // Verify that an iOS version that's not present in the map is translated
  // like the nearest major OS version.
  EXPECT_EQ(*Mapping->map(VersionTuple(10, 1), VersionTuple(), std::nullopt),
            VersionTuple(10, 0));

  // Verify that the iOS versions that are outside of the mapped version
  // range map to the min/max values passed to the `map` call.
  EXPECT_EQ(
      *Mapping->map(VersionTuple(9, 0), VersionTuple(99, 99), std::nullopt),
      VersionTuple(99, 99));
  EXPECT_EQ(
      *Mapping->map(VersionTuple(13, 0), VersionTuple(), VersionTuple(99, 99)),
      VersionTuple(99, 99));
}

TEST(DarwinSDKInfoTest, MissingKeys) {
  llvm::json::Object Obj;
  ASSERT_FALSE(DarwinSDKInfo::parseDarwinSDKSettingsJSON(&Obj));
  Obj["Version"] = "11.0";
  ASSERT_FALSE(DarwinSDKInfo::parseDarwinSDKSettingsJSON(&Obj));
}
