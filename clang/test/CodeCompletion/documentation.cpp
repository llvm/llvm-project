// Note: the run lines follow their respective tests, since line/column
// matter in this test.

/// Aaa.
void T1(float x, float y);

/// Bbb.
class T2 {
public:
  /// Ccc.
  void T3();

  int T4; ///< Ddd.
};

/// Eee.
namespace T5 {
}

void test() {

  T2 t2;
  t2.
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-brief-comments -code-completion-at=%s:21:1 %s -o - | FileCheck -check-prefix=CC1 %s
// CHECK-CC1: COMPLETION: T1 : [#void#]T1(<#float x#>, <#float y#>) : Aaa.
// CHECK-CC1: COMPLETION: T2 : T2 : Bbb.
// CHECK-CC1: COMPLETION: T5 : T5:: : Eee.

// RUN: %clang_cc1 -fsyntax-only -code-completion-brief-comments -code-completion-at=%s:23:6 %s -o - | FileCheck -check-prefix=CC2 %s
// CHECK-CC2: COMPLETION: T3 : [#void#]T3() : Ccc.
// CHECK-CC2: COMPLETION: T4 : [#int#]T4 : Ddd.
