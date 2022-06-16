// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t.dir/cache -triple x86_64-apple-macosx10.11.0 \
// RUN:   -I%t.dir/headers %t.dir/main.m -emit-llvm -o %t.dir/main.ll
// RUN: cat %t.dir/main.ll | FileCheck %s

//--- headers/a.h

__attribute__((availability(macos,introduced=10.16)))
@interface INIntent
- (instancetype)self;
@end

//--- headers/b.h

@class INIntent;

//--- headers/module.modulemap

module A {
  header "a.h"
}

module B {
  header "b.h"
}

//--- main.m

#import <a.h>
#import <b.h> // NOTE: Non attributed decl imported after one with attrs.

void F(id);

int main() {
  if (@available(macOS 11.0, *))
    F([INIntent self]);
}

// CHECK: @"OBJC_CLASS_$_INIntent" = extern_weak
