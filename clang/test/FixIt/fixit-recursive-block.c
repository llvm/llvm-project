// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -Wuninitialized -fblocks -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -Wuninitialized -fblocks -verify %s 
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -Wuninitialized -fblocks -x objective-c -fobjc-arc -DARC -verify %s

// rdar://10817031

int main() {
    void (^arc_fail)() = ^() {
#ifdef ARC
// expected-warning@-2 {{block pointer variable 'arc_fail' is null when captured by block}}
#else
// expected-warning@-4 {{block pointer variable 'arc_fail' is uninitialized when captured by block}}
#endif
// expected-note@-6 {{did you mean to use __block 'arc_fail'}}
       arc_fail(); // BOOM
    };
}
// CHECK: {8:12-8:12}:"__block "
