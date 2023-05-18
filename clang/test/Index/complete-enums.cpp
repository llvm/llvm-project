// Note: the run lines follow their respective tests, since line/column
// matter in this test.
struct X { X(); ~X(); };
enum class Color {
  Red = 17,
  Green,
  Blue
};
int Greeby();
void f(Color color) {
  switch (color) {
  case Color::Green: { X x; }
  case Color::Red;
  }
}

// RUN: c-index-test -code-completion-at=%s:12:8 -std=c++11  %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: EnumConstantDecl:{ResultType Color}{Text Color::}{TypedText Blue} (7)
// CHECK-CC1: EnumConstantDecl:{ResultType Color}{Text Color::}{TypedText Green} (7)
// CHECK-CC1: EnumConstantDecl:{ResultType Color}{Text Color::}{TypedText Red} (7)

// RUN: c-index-test -code-completion-at=%s:13:8 -std=c++11  %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: EnumConstantDecl:{ResultType Color}{Text Color::}{TypedText Blue} (7)
// CHECK-CC2-NOT: Green
// CHECK-CC2: EnumConstantDecl:{ResultType Color}{Text Color::}{TypedText Red} (7)
