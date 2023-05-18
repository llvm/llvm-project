int foo(bool x) __attribute__((enable_if(x, "")));

int test() {
  bool fffffff;
  // RUN: %clang_cc1 -std=c++11 -code-completion-at=%s:%(line+2):8 %s | FileCheck %s
  // CHECK: COMPLETION: fffffff : [#bool#]fffffff
  foo(ff
}
