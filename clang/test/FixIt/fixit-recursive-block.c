// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -Wuninitialized -fblocks -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -Wuninitialized -fblocks -verify %s 

// rdar://10817031

int main() {
    void (^arc_fail)() = ^() {  // expected-warning {{block pointer variable 'arc_fail' is uninitialized when captured by block}} \
                                // expected-note {{maybe you meant to use __block 'arc_fail'}}
       arc_fail(); // BOOM
    };
}
// CHECK: {7:12-7:12}:"__block "
