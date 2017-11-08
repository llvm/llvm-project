struct foo {
  typedef int type;

  type method(type, foo::type, ::foo::type, ::foo::foo::type);
};

namespace ns {
  struct bar {
  };

  struct baz {
  };

  int func(foo::type a, bar b, baz c);
}

typedef ns::bar bar;

int func(foo a, bar b, ns::bar c, ns::baz d);
using ns::func;

void test() {
  foo().method(0, 0, 0, 0);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:23:9 %s -o - | FileCheck %s --check-prefix=CHECK-1
  // CHECK-1: COMPLETION: method : [#type#]method(<#type#>, <#foo::type#>, <#::foo::type#>, <#::foo::foo::type#>)
  f
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:26:3 %s -o - | FileCheck %s --check-prefix=CHECK-2
  // CHECK-2: COMPLETION: func : [#int#]func(<#foo a#>, <#bar b#>, <#ns::bar c#>, <#ns::baz d#>
  // CHECK-2: COMPLETION: func : [#int#]func(<#foo::type a#>, <#bar b#>, <#baz c#>
}
