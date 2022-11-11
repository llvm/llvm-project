// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -fblocks %s 2>&1 | FileCheck %s

extern "C" void NSLog(id, ...);

void test_percent_C() {
  const unsigned short data = 'a';
  NSLog(@"%C", data);  // no-warning

  const wchar_t wchar_data = L'a';
  NSLog(@"%C", wchar_data);  // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'wchar_t'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"(unsigned short)"

  NSLog(@"%C", 0x260300);  // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'int'}}
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"(unsigned short)"

  typedef unsigned short unichar;

  NSLog(@"%C", wchar_data);  // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'wchar_t'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:16-[[@LINE-1]]:16}:"(unichar)"

  NSLog(@"%C", 0x260300);  // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'int'}}
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"(unichar)"

  NSLog(@"%C", 0.0); // expected-warning{{format specifies type 'unichar' (aka 'unsigned short') but the argument has type 'double'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%f"
  // CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"(unichar)"
}
