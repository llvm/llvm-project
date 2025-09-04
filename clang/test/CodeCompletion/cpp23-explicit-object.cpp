struct A {
  void foo(this auto&& self, int arg);
  void bar(this A self, int arg);
};

int func1() {
  A a {};
  a.
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):5 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: COMPLETION: A : A::
// CHECK-NEXT-CC1: COMPLETION: bar : [#void#]bar(<#int arg#>)
// CHECK-NEXT-CC1: COMPLETION: foo : [#void#]foo(<#int arg#>)
// CHECK-NEXT-CC1: COMPLETION: operator= : [#A &#]operator=(<#const A &#>)
// CHECK-NEXT-CC1: COMPLETION: operator= : [#A &#]operator=(<#A &&#>)
// CHECK-NEXT-CC1: COMPLETION: ~A : [#void#]~A()

struct B {
  template <typename T>
  void foo(this T&& self, int arg);
};

int func2() {
  B b {};
  b.foo();
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):9 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: OVERLOAD: [#void#]foo(int arg)

// TODO: llvm/llvm-project/146649
// This is incorrect behavior. Correct Result should be a variant of, 
// CC3: should be something like [#void#]foo(<#A self#>, <#int arg#>)
// CC4: should be something like [#void#]bar(<#A self#>, <#int arg#>)
int func3() {
  (&A::foo)
  (&A::bar)
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-3):10 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: COMPLETION: foo : [#void#]foo<<#class self:auto#>>(<#int arg#>)
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-4):10 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: COMPLETION: bar : [#void#]bar(<#int arg#>)

int func4() {
  // TODO (&A::foo)(
  (&A::bar)()
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):13 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: OVERLOAD: [#void#](<#A#>, int)

struct C {
  int member {};
  int memberFnA(int a);
  int memberFnA(this C&, float a);

  void foo(this C& self) {
    // Should not offer any members here, since 
    // it needs to be referenced through `self`.
    mem
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):8 -std=c++23 %s | FileCheck --allow-empty %s
    // CHECK-NOT: COMPLETION: member : [#int#]member
    // CHECK-NOT: COMPLETION: memberFnA : [#int#]memberFnA(<#int a#>)
    // CHECK-NOT: COMPLETION: memberFnA : [#int#]memberFnA(<#float a#>)
  }
  void bar(this C& self) {
    // should offer all results
    self.mem
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):13 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC6 %s
    // CHECK-CC6: COMPLETION: member : [#int#]member
    // CHECK-CC6: COMPLETION: memberFnA : [#int#]memberFnA(<#int a#>)
    // CHECK-CC6: COMPLETION: memberFnA : [#int#]memberFnA(<#float a#>)
  }
  void baz(this C& self) {
    [&]() {
      // Should not offer any results
      mem
      // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):10 -std=c++23 %s | FileCheck --allow-empty %s
      // CHECK-NOT: COMPLETION: member : [#int#]member
      // CHECK-NOT: COMPLETION: memberFnA : [#int#]memberFnA(<#int a#>)
      // CHECK-NOT: COMPLETION: memberFnA : [#int#]memberFnA(<#float a#>)
    }();
  }
};


struct S {
  void foo1(int a);
  void foo2(int a) const;
  void foo2(this const S& self, float a);  
  void foo3(this const S& self, int a);
  void foo4(this S& self, int a);
};

void S::foo1(int a) {
  this->;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):9 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: COMPLETION: foo1 : [#void#]foo1(<#int a#>)
// CHECK-CC7: COMPLETION: foo2 : [#void#]foo2(<#int a#>)[# const#]
// CHECK-CC7: COMPLETION: foo2 : [#void#]foo2(<#float a#>)[# const#]
// CHECK-CC7: COMPLETION: foo3 : [#void#]foo3(<#int a#>)[# const#]
// CHECK-CC7: COMPLETION: foo4 : [#void#]foo4(<#int a#>)
}

void S::foo2(int a) const {
  this->;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):9 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: COMPLETION: foo2 : [#void#]foo2(<#int a#>)[# const#]
// CHECK-CC8: COMPLETION: foo2 : [#void#]foo2(<#float a#>)[# const#]
// CHECK-CC8: COMPLETION: foo3 : [#void#]foo3(<#int a#>)[# const#]
}

void S::foo3(this const S& self, int a) {
  self.;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):8 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9: COMPLETION: foo2 : [#void#]foo2(<#int a#>)[# const#]
// CHECK-CC9: COMPLETION: foo2 : [#void#]foo2(<#float a#>)[# const#]
// CHECK-CC9: COMPLETION: foo3 : [#void#]foo3(<#int a#>)[# const#]
}

void S::foo4(this S& self, int a) {
  self.;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):8 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC10 %s
// CHECK-CC10: COMPLETION: foo1 : [#void#]foo1(<#int a#>)
// CHECK-CC10: COMPLETION: foo2 : [#void#]foo2(<#int a#>)[# const#]
// CHECK-CC10: COMPLETION: foo2 : [#void#]foo2(<#float a#>)[# const#]
// CHECK-CC10: COMPLETION: foo3 : [#void#]foo3(<#int a#>)[# const#]
// CHECK-CC10: COMPLETION: foo4 : [#void#]foo4(<#int a#>)
}

void test1(S s) {
  s.;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):5 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC11 %s
// CHECK-CC11: COMPLETION: foo1 : [#void#]foo1(<#int a#>)
// CHECK-CC11: COMPLETION: foo2 : [#void#]foo2(<#int a#>)[# const#]
// CHECK-CC11: COMPLETION: foo2 : [#void#]foo2(<#float a#>)[# const#]
// CHECK-CC11: COMPLETION: foo3 : [#void#]foo3(<#int a#>)[# const#]
// CHECK-CC11: COMPLETION: foo4 : [#void#]foo4(<#int a#>)
}

void test2(const S s) {
  s.;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):5 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC12 %s
// CHECK-CC12: COMPLETION: foo2 : [#void#]foo2(<#int a#>)[# const#]
// CHECK-CC12: COMPLETION: foo2 : [#void#]foo2(<#float a#>)[# const#]
// CHECK-CC12: COMPLETION: foo3 : [#void#]foo3(<#int a#>)[# const#]
}

void test3(S s) {
  s.foo2();
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):10 -std=c++23 %s | FileCheck -check-prefix=CHECK-CC13 %s
// CHECK-CC13: OVERLOAD: [#void#]foo2(<#int a#>)
// CHECK-CC13: OVERLOAD: [#void#]foo2(float a)
// TODO: foo2 should be OVERLOAD: [#void#]foo2(<#float a#>)
}
