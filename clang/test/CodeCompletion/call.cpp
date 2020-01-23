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
