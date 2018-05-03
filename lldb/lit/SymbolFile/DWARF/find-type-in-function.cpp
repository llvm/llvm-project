// REQUIRES: lld

// XFAIL: *

// RUN: clang %s -g -c -o %t.o --target=x86_64-pc-linux
// RUN: ld.lld %t.o -o %t
// RUN: lldb-test symbols --name=foo --find=type %t | \
// RUN:   FileCheck --check-prefix=NAME %s

// Lookup for "foo" should find either both "struct foo" types or just the
// global one. Right now, it finds the definition inside bar(), which is
// definitely wrong.

// NAME: Found 2 types:
struct foo {};
// NAME-DAG: name = "foo", {{.*}} decl = find-type-in-function.cpp:[[@LINE-1]]

void bar() {
  struct foo {};
// NAME-DAG: name = "foo", {{.*}} decl = find-type-in-function.cpp:[[@LINE-1]]
  foo a;
}

extern "C" void _start(foo) {}
