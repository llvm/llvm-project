// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol NSCopying @end

@interface NSString <NSCopying>
@end

void takeId(id test) {}

void takeCopyableId(id<NSCopying> test) {}

id<NSCopying> Test (void) {
  NSString const *constantString = @"Test";
  takeId(constantString);
  takeCopyableId(constantString);
  id ID = constantString;
  id<NSCopying> IDQNSCopying = constantString;
  return constantString;
}
