// Note: the run lines follow their respective tests, since line/column
// matter in this test.
void f(float x, float y);
void f(int i, int j, int k);
struct X { };
void f(X);
namespace N {
  struct Y { 
    Y(int = 0); 
    
    operator int() const;
  };
  void f(Y y, int ZZ);
}
typedef N::Y Y;
void f();

void test() {
  f(Y(), 0, 0);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:19:9 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: f(Y y, <#int ZZ#>)
  // CHECK-CC1-NEXT: f(int i, <#int j#>, int k)
  // CHECK-CC1-NEXT: f(float x, <#float y#>)
  // CHECK-CC1: COMPLETION: Pattern : dynamic_cast<<#type#>>(<#expression#>)
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:19:13 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2-NOT: f(Y y, int ZZ)
  // CHECK-CC2: f(int i, int j, <#int k#>)
  f({}, 0, 0);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:28:7 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: OVERLOAD: [#void#]f()
  // CHECK-CC3-NEXT: OVERLOAD: [#void#]f(<#X#>)
  // CHECK-CC3-NEXT: OVERLOAD: [#void#]f(<#int i#>, int j, int k)
  // CHECK-CC3-NEXT: OVERLOAD: [#void#]f(<#float x#>, float y)
}

void f(int, int, int, int);
template <typename T>
void foo(T t) {
  f(t, t, t);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:39:5 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
  // CHECK-CC4: f()
  // CHECK-CC4-NEXT: f(<#X#>)
  // CHECK-CC4-NEXT: f(<#int i#>, int j, int k)
  // CHECK-CC4-NEXT: f(<#float x#>, float y)
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:39:8 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
  // CHECK-CC5-NOT: f()
  // CHECK-CC5: f(int i, <#int j#>, int k)
  // CHECK-CC5-NEXT: f(float x, <#float y#>)
  f(5, t, t);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:49:11 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
  // CHECK-CC6-NOT: f(float x, float y)
  // CHECK-CC6: f(int, int, <#int#>, int)
  // CHECK-CC6-NEXT: f(int i, int j, <#int k#>)
}
