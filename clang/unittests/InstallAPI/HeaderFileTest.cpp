//===- unittests/InstallAPI/HeaderFile.cpp - HeaderFile Test --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/HeaderFile.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace clang::installapi;

namespace HeaderFileTests {

TEST(HeaderFile, FrameworkIncludes) {
  const char *Path = "/System/Library/Frameworks/Foo.framework/Headers/Foo.h";
  std::optional<std::string> IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Foo/Foo.h");

  Path = "/System/Library/Frameworks/Foo.framework/Frameworks/Bar.framework/"
         "Headers/SimpleBar.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Bar/SimpleBar.h");

  Path = "/tmp/Foo.framework/Versions/A/Headers/SimpleFoo.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Foo/SimpleFoo.h");

  Path = "/System/Library/PrivateFrameworks/Foo.framework/Headers/Foo.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Foo/Foo.h");

  Path = "/AppleInternal/Developer/Library/Frameworks/"
         "HelloFramework.framework/Headers/HelloFramework.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "HelloFramework/HelloFramework.h");

  Path = "/tmp/BuildProducts/Foo.framework/Versions/A/"
         "PrivateHeaders/Foo+Private.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Foo/Foo+Private.h");

  Path = "/Applications/Xcode.app/Contents/Developer/SDKS/MacOS.sdk/System/"
         "Library/Frameworks/Foo.framework/PrivateHeaders/Foo_Private.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Foo/Foo_Private.h");

  Path =
      "/System/Library/PrivateFrameworks/Foo.framework/PrivateHeaders/Foo.hpp";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Foo/Foo.hpp");

  Path = "/Applications/Xcode.app/Contents/Developer/SDKS/MacOS.sdk/System/"
         "Library/Frameworks/Foo.framework/Headers/BarDir/Bar.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "Foo/BarDir/Bar.h");
}

TEST(HeaderFile, DylibIncludes) {
  const char *Path = "/usr/include/foo.h";
  std::optional<std::string> IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "foo.h");

  Path = "/tmp/BuildProducts/usr/include/a/A.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "a/A.h");

  Path = "/Applications/Xcode.app/Contents/Developer/SDKS/MacOS.sdk/"
         "usr/include/simd/types.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "simd/types.h");

  Path = "/usr/local/include/hidden/A.h";
  IncludeName = createIncludeHeaderName(Path);
  EXPECT_TRUE(IncludeName.has_value());
  EXPECT_STREQ(IncludeName.value().c_str(), "hidden/A.h");
}
} // namespace HeaderFileTests
