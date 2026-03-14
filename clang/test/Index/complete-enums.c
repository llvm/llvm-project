// Note: the run lines follow their respective tests, since line/column
// matter in this test.

enum __attribute__((deprecated)) Color {
  Color_Red = 17,
  Color_Green,
  Color_Blue
};
int Greeby();
void f(enum Color color) {
  switch (color) {
  case Red:
  }
}

// RUN: c-index-test -code-completion-at=%s:11:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: EnumConstantDecl:{ResultType enum Color}{TypedText Color_Red} (65) (deprecated)

// RUN: c-index-test -code-completion-at=%s:12:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: EnumConstantDecl:{ResultType enum Color}{TypedText Color_Blue} (7) (deprecated)
// CHECK-CC2-NEXT: EnumConstantDecl:{ResultType enum Color}{TypedText Color_Green} (7) (deprecated)
// CHECK-CC2-NEXT: EnumConstantDecl:{ResultType enum Color}{TypedText Color_Red} (7) (deprecated)
