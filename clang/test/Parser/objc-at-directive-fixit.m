// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.10.0 -verify -fobjc-exceptions %s
// RUN: not %clang_cc1 -fsyntax-only -triple x86_64-apple-macosx10.10.0 -fdiagnostics-parseable-fixits -fobjc-exceptions %s 2>&1 | FileCheck %s

// rdar://19669565

void bar(int x);

void f() {
  @try { }
  @finally { }
  @autoreleasepool { }

  // Provide a fixit when we are parsing a standalone statement
  @tr { }; // expected-error {{unexpected '@' in program}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:6}:"try"
  @finaly { }; // expected-error {{unexpected '@' in program}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:10}:"finally"
  @autorelpool { }; // expected-error {{unexpected '@' in program}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:4-[[@LINE-1]]:15}:"autoreleasepool"

  // Ensure that no fixit is given when parsing expressions
  // CHECK-NOT: fix-it
  id thing = @autoreleasepool { }; // expected-error {{unexpected '@' in program}}
  (void)@tr { }; // expected-error {{unexpected '@' in program}}
  bar(@final { }); // expected-error {{unexpected '@' in program}}
  for(@auto;;) { } // expected-error {{unexpected '@' in program}}
  [@try]; // expected-error {{unexpected '@' in program}}
}
