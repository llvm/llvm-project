template <class T>
struct function {
};


void test() {
  void (*x)(int, double) = nullptr;

  function<void(int, double)> y = {};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-3):28 %s -o - | FileCheck -check-prefix=CHECK-1 %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-2):35 %s -o - | FileCheck -check-prefix=CHECK-1 %s
  // CHECK-1: COMPLETION: Pattern : [<#=#>](int <#parameter#>, double <#parameter#>) { <#body#> }

  // == Placeholders for suffix types must be placed properly.
  function<void(void(*)(int))> z = {};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):36 %s -o - | FileCheck -check-prefix=CHECK-2 %s
  // CHECK-2: COMPLETION: Pattern : [<#=#>](void (* <#parameter#>)(int)) { <#body#> }

  // == No need for a parameter list if function has no parameters.
  function<void()> a = {};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):24 %s -o - | FileCheck -check-prefix=CHECK-3 %s
  // CHECK-3: COMPLETION: Pattern : [<#=#>] { <#body#> }
}

template <class T, class Allocator = int>
struct vector {};

void test2() {
  // == Try to preserve types as written.
  function<void(vector<int>)> a = {};

  using function_typedef = function<void(vector<int>)>;
  function_typedef b = {};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-4):35 %s -o - | FileCheck -check-prefix=CHECK-4 %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-2):24 %s -o - | FileCheck -check-prefix=CHECK-4 %s
  // CHECK-4: COMPLETION: Pattern : [<#=#>](vector<int> <#parameter#>) { <#body#> }
}

// Check another common function wrapper name.
template <class T> struct unique_function {};

void test3() {
  unique_function<void()> a = {};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):31 %s -o - | FileCheck -check-prefix=CHECK-5 %s
  // CHECK-5: COMPLETION: Pattern : [<#=#>] { <#body#> }
}

template <class T, class U> struct weird_function {};
void test4() {
  weird_function<void(), int> b = {};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):35 %s -o - | FileCheck -check-prefix=CHECK-6 %s
  // CHECK-6-NOT: COMPLETION: Pattern : [<#=
}

void test5() {
  // Completions are only added when -code-completion-patterns are enabled.
  function<void()> b = {};
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):24 %s -o - | FileCheck -check-prefix=CHECK-7 %s
  // CHECK-7: COMPLETION: Pattern : [<#=
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-3):24 %s -o - | FileCheck -check-prefix=CHECK-8 %s
  // CHECK-8-NOT: COMPLETION: Pattern : [<#=
}

void test6() {
  auto my_lambda = [&](int a, double &b) { return 1.f; };
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):58 %s -o - | FileCheck -check-prefix=CHECK-9 %s
  // CHECK-9: [#float#]my_lambda(<#int a#>, <#double &b#>)[# const#]
}

void test7() {
  auto generic_lambda = [&](auto a, const auto &b) { return a + b; };
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):70 %s -o - | FileCheck -check-prefix=CHECK-10 %s
  // CHECK-10: [#auto#]generic_lambda(<#auto a#>, <#const auto &b#>)[# const#]
}
