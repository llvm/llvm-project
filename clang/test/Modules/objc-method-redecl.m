// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x objective-c-header -emit-pch %S/Inputs/objc-method-redecl.h -o %t.pch -Wno-objc-root-class
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -x objective-c -include-pch %t.pch %s -verify -Wno-objc-root-class
// expected-no-diagnostics

@implementation T
- (void)test {
}
@end
