// RUN: %check_clang_tidy %s modernize-redundant-void-arg %t -- -- -fdelayed-template-parsing

int foo(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
// CHECK-FIXES: {{^}}int foo() {{{$}}
    return 0;
}

template <class T>
struct MyFoo {
  int foo(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
// CHECK-FIXES: {{^}}  int foo() {{{$}}
    return 0;
  }
};
// Explicit instantiation.
template class MyFoo<int>;

template <class T>
struct MyBar {
  // This declaration isn't instantiated and won't be parsed 'delayed-template-parsing'.
  int foo(void) {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant void argument list in function definition [modernize-redundant-void-arg]
// CHECK-FIXES: {{^}}  int foo() {{{$}}
    return 0;
  }
};
