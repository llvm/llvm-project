template <typename T, typename... Args>
void fun(T x, Args... args) {}

void f() {
  fun(1, 2, 3, 4);
  // The results are quite awkward here, but it's the best we can do for now.
  // Tools, including clangd, can unexpand "args" when showing this to the user.
  // The important thing is that we provide OVERLOAD signature in all those cases.
  //
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:5:7 %s -o - | FileCheck --check-prefix=CHECK-1 %s
  // CHECK-1: OVERLOAD: [#void#]fun(<#T x#>, Args args...)
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:5:10 %s -o - | FileCheck --check-prefix=CHECK-2 %s
  // CHECK-2: OVERLOAD: [#void#]fun(int x)
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:5:13 %s -o - | FileCheck --check-prefix=CHECK-3 %s
  // CHECK-3: OVERLOAD: [#void#]fun(int x, int args)
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:5:16 %s -o - | FileCheck --check-prefix=CHECK-4 %s
  // CHECK-4: OVERLOAD: [#void#]fun(int x, int args, int args)
}
