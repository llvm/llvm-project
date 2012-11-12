// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpTraversal -analyzer-max-loop 4 -std=c++11 %s | FileCheck -check-prefix=DFS %s

int a();
int b();
int c();

int work();

void test(id input) {
  if (a()) {
    if (a())
      b();
    else
      c();
  } else {
    if (b())
      a();
    else
      c();
  }

  if (a())
    work();
}

void testLoops(id input) {
  while (a()) {
    work();
    work();
    work();
  }

  for (int i = 0; i != b(); ++i) {
    work();
  }

  for (id x in input) {
    work();
    work();
    work();
  }

  int z[] = {1,2,3};
  for (int y : z) {
    work();
    work();
    work();
  }
}

// This ordering assumes that false cases happen before the true cases.

// DFS:27 WhileStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:27 WhileStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:27 WhileStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:27 WhileStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:33 ForStmt
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:44 CXXForRangeStmt
// DFS-next:--END PATH--
// DFS-next:37 ObjCForCollectionStmt
// DFS-next:10 IfStmt
// DFS-next:16 IfStmt
// DFS-next:22 IfStmt
// DFS-next:--END PATH--
// DFS-next:--END PATH--
// DFS-next:22 IfStmt
// DFS-next:--END PATH--
// DFS-next:--END PATH--
// DFS-next:11 IfStmt
// DFS-next:22 IfStmt
// DFS-next:--END PATH--
// DFS-next:--END PATH--
// DFS-next:22 IfStmt
// DFS-next:--END PATH--
// DFS-next:--END PATH--

