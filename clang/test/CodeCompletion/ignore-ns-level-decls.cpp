namespace ns {
  struct bar {
  };

  struct baz {
  };

  int func(int a, bar b, baz c);
}

void test() {
  ns::
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):7 %s -o - | FileCheck %s --check-prefix=CHECK-1
// CHECK-1-DAG: COMPLETION: bar : bar
// CHECK-1-DAG: COMPLETION: baz : baz
// CHECK-1-DAG: COMPLETION: func : [#int#]func(<#int a#>, <#bar b#>, <#baz c#>)

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-6):7 -no-code-completion-ns-level-decls %s -o - | FileCheck %s --allow-empty --check-prefix=CHECK-EMPTY
// CHECK-EMPTY-NOT: COMPLETION: bar : bar
// CHECK-EMPTY: {{^}}{{$}}
}
