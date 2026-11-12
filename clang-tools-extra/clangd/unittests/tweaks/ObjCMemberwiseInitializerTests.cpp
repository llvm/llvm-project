//===-- ObjCMemberwiseInitializerTests.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestTU.h"
#include "TweakTesting.h"
#include "gmock/gmock-matchers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace {

TWEAK_TEST(ObjCMemberwiseInitializer);

TEST_F(ObjCMemberwiseInitializerTest, TestAvailability) {
  FileName = "TestTU.m";

  // Ensure the action can't be triggered since arc is disabled.
  EXPECT_UNAVAILABLE(R"cpp(
    @interface Fo^o
    @end
  )cpp");

  ExtraArgs.push_back("-fobjc-runtime=macosx");
  ExtraArgs.push_back("-fobjc-arc");

  // Ensure the action can be initiated on the interface and implementation,
  // but not on the forward declaration.
  EXPECT_AVAILABLE(R"cpp(
    @interface Fo^o
    @end
  )cpp");
  EXPECT_AVAILABLE(R"cpp(
    @interface Foo
    @end

    @implementation F^oo
    @end
  )cpp");
  EXPECT_UNAVAILABLE("@class Fo^o;");

  // Ensure that the action can be triggered on ivars and properties,
  // including selecting both.
  EXPECT_AVAILABLE(R"cpp(
    @interface Foo {
      id _fi^eld;
    }
    @end
  )cpp");
  EXPECT_AVAILABLE(R"cpp(
    @interface Foo
    @property(nonatomic) id fi^eld;
    @end
  )cpp");
  EXPECT_AVAILABLE(R"cpp(
    @interface Foo {
      id _fi^eld;
    }
    @property(nonatomic) id pr^op;
    @end
  )cpp");

  // Ensure that the action can't be triggered on property synthesis
  // and methods.
  EXPECT_UNAVAILABLE(R"cpp(
    @interface Foo
    @property(nonatomic) id prop;
    @end

    @implementation Foo
    @dynamic pr^op;
    @end
  )cpp");
  EXPECT_UNAVAILABLE(R"cpp(
    @interface Foo
    @end

    @implementation Foo
    - (void)fo^o {}
    @end
  )cpp");
}

TEST_F(ObjCMemberwiseInitializerTest, Test) {
  FileName = "TestTU.m";
  ExtraArgs.push_back("-fobjc-runtime=macosx");
  ExtraArgs.push_back("-fobjc-arc");

  const char *Input = R"cpp(
@interface Foo {
  id [[_field;
}
@property(nonatomic) id prop]];
@property(nonatomic) id notSelected;
@end)cpp";
  const char *Output = R"cpp(
@interface Foo {
  id _field;
}
@property(nonatomic) id prop;
@property(nonatomic) id notSelected;
- (instancetype)initWithField:(id)field prop:(id)prop;

@end)cpp";
  EXPECT_EQ(apply(Input), Output);

  Input = R"cpp(
@interface Foo
@property(nonatomic, nullable) id somePrettyLongPropertyName;
@property(nonatomic, nonnull) id someReallyLongPropertyName;
@end

@implementation F^oo

- (instancetype)init {
  return self;
}

@end)cpp";
  Output = R"cpp(
@interface Foo
@property(nonatomic, nullable) id somePrettyLongPropertyName;
@property(nonatomic, nonnull) id someReallyLongPropertyName;
- (instancetype)initWithSomePrettyLongPropertyName:(nullable id)somePrettyLongPropertyName someReallyLongPropertyName:(nonnull id)someReallyLongPropertyName;

@end

@implementation Foo

- (instancetype)init {
  return self;
}

- (instancetype)initWithSomePrettyLongPropertyName:(nullable id)somePrettyLongPropertyName someReallyLongPropertyName:(nonnull id)someReallyLongPropertyName {
  self = [super init];
  if (self) {
    _somePrettyLongPropertyName = somePrettyLongPropertyName;
    _someReallyLongPropertyName = someReallyLongPropertyName;
  }
  return self;
}

@end)cpp";
  EXPECT_EQ(apply(Input), Output);
}

} // namespace
} // namespace clangd
} // namespace clang
