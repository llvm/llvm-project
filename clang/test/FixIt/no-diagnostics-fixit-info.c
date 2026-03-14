// RUN: not %clang_cc1 %s 2>&1 | FileCheck -check-prefix=WITH-FIXIT %s
// RUN: not %clang_cc1 -fno-diagnostics-fixit-info %s 2>&1 | FileCheck -check-prefix=WITHOUT-FIXIT %s

struct Foo {
  int x;
}
// WITH-FIXIT: error: expected ';' after struct
// WITH-FIXIT-NEXT: }
// WITH-FIXIT-NEXT:  ^
// WITH-FIXIT-NEXT:  ;

// WITHOUT-FIXIT: error: expected ';' after struct
// WITHOUT-FIXIT-NEXT: }
// WITHOUT-FIXIT-NEXT: ^
// WITHOUT-FIXIT-NOT: ;

