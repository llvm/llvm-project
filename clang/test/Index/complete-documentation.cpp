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

struct T6 {
 /// \brief Fff.
 void T7();

 /// \brief Ggg.
 void T8();
};

void T6::T7() {
}

void test1() {

  T2 t2;
  t2.T4;

  T6 t6;
  t6.T8();
}

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:32:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: FunctionDecl:{ResultType void}{TypedText T1}{{.*}}(brief comment: Aaa.)
// CHECK-CC1: ClassDecl:{TypedText T2}{{.*}}(brief comment: Bbb.)
// CHECK-CC1: Namespace:{TypedText T5}{{.*}}(brief comment: Eee.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:34:6 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: CXXMethod:{ResultType void}{TypedText T3}{{.*}}(brief comment: Ccc.)
// CHECK-CC2: FieldDecl:{ResultType int}{TypedText T4}{{.*}}(brief comment: Ddd.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:37:6 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: CXXMethod:{ResultType void}{TypedText T7}{LeftParen (}{RightParen )} (34)(brief comment: Fff.)
// CHECK-CC3: CXXMethod:{ResultType void}{TypedText T8}{LeftParen (}{RightParen )} (34)(brief comment: Ggg.)
