// RUN: %clang_cc1 -arcmt-check -fsyntax-only -fobjc-arc -x objective-c %s

@protocol NSObject
- (oneway void)release;
@end

void test1(id p) {
  [p release];
}
