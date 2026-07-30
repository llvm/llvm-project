// RUN: c-index-test -test-load-source-usrs local -std=c++20 -- %s | FileCheck %s

// Check USRs of template specializations with structural NTTP values.

template <auto> struct Tpl{};

struct {
  int n;
} s;

void fn1(Tpl<1.5>);
// CHECK: fn1#$@S@Tpl>#Sd[[#HASH:]]#
void fn2(Tpl<1.7>);
// CHECK-NOT: [[#HASH]]
void fn1(Tpl<1.5>) {}
// CHECK: fn1#$@S@Tpl>#Sd[[#HASH]]#

void fn(Tpl<&s.n>);
// CHECK: #S*I[[#HASH:]]#
void fn(Tpl<(void*)&s.n>);
// CHECK: #S*v[[#HASH]]#
void fn(Tpl<&s.n>) {}
// CHECK: #S*I[[#HASH]]#
