// Test this without pch.
// RUN: %clang_cc1 -include %S/objc_import.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -x objective-c -emit-pch -o %t %S/objc_import.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

// expected-no-diagnostics

#import "objc_import.h"

void func(void) {
 TestPCH *xx;

 xx = [TestPCH alloc];
 [xx instMethod];
}

// rdar://14112291
@class NewID1;
void foo1(NewID1 *p);
void bar1(OldID1 *p) {
  foo1(p);
}
@class NewID2;
void foo2(NewID2 *p) {
  [p meth];
}
void bar2(OldID2 *p) {
  foo2(p);
  [p meth];
}
