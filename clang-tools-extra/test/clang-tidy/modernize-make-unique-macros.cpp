// RUN: %check_clang_tidy %s modernize-make-unique %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-make-unique.IgnoreMacros, value: 0}]}" \
// RUN:   -- -std=c++14  -I%S/Inputs/modernize-smart-ptr

#include "unique_ptr.h"

class Foo {};
class Bar {};
#define DEFINE(...) __VA_ARGS__
// CHECK-FIXES: {{^}}#define DEFINE(...) __VA_ARGS__{{$}}
template<typename T>
void g2(std::unique_ptr<Foo> *t) {
  DEFINE(
  // CHECK-FIXES: {{^ *}}DEFINE({{$}}
      auto p = std::unique_ptr<Foo>(new Foo);
      // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use std::make_unique instead
      // CHECK-FIXES: {{^ *}}auto p = std::unique_ptr<Foo>(new Foo);{{$}}
      t->reset(new Foo);
      // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::make_unique instead
      // CHECK-FIXES: {{^ *}}t->reset(new Foo);{{$}}
      );
      // CHECK-FIXES: {{^ *}});{{$}}
}
void macro() {
  std::unique_ptr<Foo> *t;
  g2<Bar>(t);
}
#undef DEFINE
