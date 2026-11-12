// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface PBXBuildSettingsDictionary
{
  int i;
}
@end

@interface XCConditionalBuildSettingsDictionary : PBXBuildSettingsDictionary
{
}
@end

@implementation PBXBuildSettingsDictionary

- (XCConditionalBuildSettingsDictionary *)conditionalDictionaryForConditionSet
{
  return i ? self : (id)0;
}

- (XCConditionalBuildSettingsDictionary *)conditionalDictionaryForConditionSet2
{
  return i ? (id)0 : self;
}
@end


