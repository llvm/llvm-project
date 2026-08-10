// RUN: rm -rf %t.cache
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t.cache \
// RUN:   %s -fsyntax-only -F %S/Inputs -Wincomplete-umbrella -verify

// expected-no-diagnostics

#import <FooFramework/Foo.h>

@implementation Foo
@end
