// This test is line- and column-sensitive. See below for run lines.

int global;

struct X {
  static int member;
  void f(int zed) {
    int local;
    static int local_static;
    [=] {
      int inner_local;
      [local, this, inner_local] {
      }
    }();
  }
};


// RUN: c-index-test -code-completion-at=%s:12:8 -std=c++11 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: VarDecl:{ResultType int}{TypedText inner_local} (34)
// CHECK-CC1-NEXT: VarDecl:{ResultType int}{TypedText local} (34)
// CHECK-CC1-NEXT: Pattern:{ResultType X *}{TypedText this} (40)
// CHECK-CC1-NEXT: ParmDecl:{ResultType int}{TypedText zed} (34)

// RUN: c-index-test -code-completion-at=%s:12:15 -std=c++11 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: VarDecl:{ResultType int}{TypedText inner_local} (34)
// CHECK-CC2-NEXT: Pattern:{ResultType X *}{TypedText this} (40)
// CHECK-CC2-NEXT: ParmDecl:{ResultType int}{TypedText zed} (34)

// RUN: c-index-test -code-completion-at=%s:12:21 -std=c++11 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: VarDecl:{ResultType int}{TypedText inner_local} (34)
// CHECK-CC3-NEXT: ParmDecl:{ResultType int}{TypedText zed} (34)

// RUN: c-index-test -code-completion-at=%s:12:8 -x objective-c++ -std=c++11 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: TypedefDecl:{TypedText Class} (50)
// CHECK-CC4: TypedefDecl:{TypedText id} (50)
// CHECK-CC4: VarDecl:{ResultType int}{TypedText inner_local} (34)
// CHECK-CC4: VarDecl:{ResultType int}{TypedText local} (34)
// CHECK-CC4: Pattern:{ResultType X *}{TypedText this} (40)
// CHECK-CC4: ParmDecl:{ResultType int}{TypedText zed} (34)

// RUN: c-index-test -code-completion-at=%s:12:15 -x objective-c++ -std=c++11 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:12:21 -x objective-c++ -std=c++11 %s | FileCheck -check-prefix=CHECK-CC3 %s
