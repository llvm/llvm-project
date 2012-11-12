// Note: the run lines follow their respective tests, since line/column numbers
// matter in this test.

/// This is T1.
template<typename T>
void T1(T t) { }

/// This is T2.
template<typename T>
void T2(T t) { }

/// This is T2<int>.
template<>
void T2(int t) { }

void test_CC1() {

}

// Check that implicit instantiations of class templates and members pick up
// comments from class templates and specializations.

/// This is T3.
template<typename T>
class T3 {
public:
  /// This is T4.
  static void T4();

  /// This is T5.
  static int T5;

  /// This is T6.
  void T6();

  /// This is T7.
  int T7;

  /// This is T8.
  class T8 {};

  /// This is T9.
  enum T9 {
    /// This is T10.
    T10
  };

  /// This is T11.
  template<typename U>
  void T11(U t) {}

  typedef T3<double> T12;
};

void test_CC2_CC3_CC4() {
  T3<int>::T4();
  T3<int> t3;
  t3.T6();
  T3<int>::T8 t8;
}

/// This is T100.
template<typename T, typename U>
class T100 {
};

/// This is T100<int, T>.
template<typename T>
class T100<int, T> {
public:
  /// This is T101.
  static void T101();

  /// This is T102.
  static int T102;

  /// This is T103.
  void T103();

  /// This is T104.
  int T104;

  /// This is T105.
  class T105 {};

  /// This is T106.
  enum T106 {
    /// This is T107.
    T107
  };

  /// This is T108.
  template<typename U>
  void T108(U t) {}

  typedef T100<double, T> T109;

  typedef T100<double, double> T110;
};

void test_CC5_CC6_CC7() {
  T100<int, long>::T101();
  T100<int, long> t100;
  t100.T103();
  T100<int, long>::T105 t105;
}

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:17:1 %s | FileCheck -check-prefix=CC1 %s
// CHECK-CC1: FunctionTemplate:{ResultType void}{TypedText T1}{{.*}}(brief comment: This is T1.)
// CHECK-CC1: FunctionTemplate:{ResultType void}{TypedText T2}{{.*}}(brief comment: This is T2.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:56:12 %s | FileCheck -check-prefix=CC2 %s
// CHECK-CC2: CXXMethod:{ResultType void}{TypedText T4}{{.*}}(brief comment: This is T4.)
// CHECK-CC2: VarDecl:{ResultType int}{TypedText T5}{{.*}}(brief comment: This is T5.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:58:6 %s | FileCheck -check-prefix=CC3 %s
// CHECK-CC3: FunctionTemplate:{ResultType void}{TypedText T11}{{.*}}(brief comment: This is T11.)
// CHECK-CC3: CXXMethod:{ResultType void}{TypedText T6}{{.*}}(brief comment: This is T6.)
// CHECK-CC3: FieldDecl:{ResultType int}{TypedText T7}{{.*}}(brief comment: This is T7.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:59:12 %s | FileCheck -check-prefix=CC4 %s
// CHECK-CC4: EnumConstantDecl:{ResultType T3<int>::T9}{TypedText T10}{{.*}}(brief comment: This is T10.)
// FIXME: after we implement propagating comments through typedefs, this
// typedef for implicit instantiation should pick up the documentation
// comment from class template.
// CHECK-CC4: TypedefDecl:{TypedText T12}
// CHECK-CC4-SHOULD-BE: TypedefDecl:{TypedText T12}{{.*}}(brief comment: This is T3.)
// CHECK-CC4: ClassDecl:{TypedText T8}{{.*}}(brief comment: This is T8.)
// CHECK-CC4: EnumDecl:{TypedText T9}{{.*}}(brief comment: This is T9.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:102:20 %s | FileCheck -check-prefix=CC5 %s
// CHECK-CC5: CXXMethod:{ResultType void}{TypedText T101}{{.*}}(brief comment: This is T101.)
// CHECK-CC5: VarDecl:{ResultType int}{TypedText T102}{{.*}}(brief comment: This is T102.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:104:8 %s | FileCheck -check-prefix=CC6 %s
// CHECK-CC6: CXXMethod:{ResultType void}{TypedText T103}{{.*}}(brief comment: This is T103.)
// CHECK-CC6: FieldDecl:{ResultType int}{TypedText T104}{{.*}}(brief comment: This is T104.)
// CHECK-CC6: FunctionTemplate:{ResultType void}{TypedText T108}{{.*}}(brief comment: This is T108.)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:105:20 %s | FileCheck -check-prefix=CC7 %s
// CHECK-CC7: ClassDecl:{TypedText T105}{{.*}}(brief comment: This is T105.)
// CHECK-CC7: EnumDecl:{TypedText T106}{{.*}}(brief comment: This is T106.)
// CHECK-CC7: EnumConstantDecl:{ResultType T100<int, long>::T106}{TypedText T107}{{.*}}(brief comment: This is T107.)
// FIXME: after we implement propagating comments through typedefs, these two
// typedefs for implicit instantiations should pick up the documentation
// comment from class template.
// CHECK-CC7: TypedefDecl:{TypedText T109}
// CHECK-CC7: TypedefDecl:{TypedText T110}
// CHECK-CC7-SHOULD-BE: TypedefDecl:{TypedText T109}{{.*}}(brief comment: This is T100.)
// CHECK-CC7-SHOULD-BE: TypedefDecl:{TypedText T110}{{.*}}(brief comment: This is T100.)

