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

TEST(DarwinSDKInfoTest, ParseAndTestMapping) {
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
  EXPECT_EQ(*Mapping->map(VersionTuple(10, 15), VersionTuple(), None),
            VersionTuple(13, 1));
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 0), VersionTuple(), None),
            VersionTuple(14, 0));
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 2), VersionTuple(), None),
            VersionTuple(14, 2));

  // Verify that a macOS version that's not present in the map is translated
  // like the nearest major OS version.
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 1), VersionTuple(), None),
            VersionTuple(14, 0));

  // Verify that the macOS versions that are outside of the mapped version
  // range map to the min/max values passed to the `map` call.
  EXPECT_EQ(*Mapping->map(VersionTuple(10, 14), VersionTuple(99, 99), None),
            VersionTuple(99, 99));
  EXPECT_EQ(
      *Mapping->map(VersionTuple(11, 5), VersionTuple(), VersionTuple(99, 99)),
      VersionTuple(99, 99));
  EXPECT_EQ(*Mapping->map(VersionTuple(11, 5), VersionTuple(99, 98),
                          VersionTuple(99, 99)),
            VersionTuple(99, 99));
}

TEST(DarwinSDKInfoTest, MissingKeys) {
  llvm::json::Object Obj;
  ASSERT_FALSE(DarwinSDKInfo::parseDarwinSDKSettingsJSON(&Obj));
  Obj["Version"] = "11.0";
  ASSERT_FALSE(DarwinSDKInfo::parseDarwinSDKSettingsJSON(&Obj));
}
