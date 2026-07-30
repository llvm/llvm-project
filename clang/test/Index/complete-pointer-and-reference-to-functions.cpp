// Note: the run lines follow their respective tests, since line/column
// matter in this test.

template<class T> void (&foo(T))(T);
template<class T> void (*bar(T))(T);
template<class T> void (&baz(T))(T, ...);
template<class T> void (*fiz(T))(T, ...);

int main() {
  foo(42)(42);
  bar(42)(42);
  baz(42)(42, 42, 42);
  fiz(42)(42, 42, 42);
}

// RUN: c-index-test -code-completion-at=%s:10:11 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: OverloadCandidate:{ResultType void}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC1: Completion contexts:
// CHECK-CC1-NEXT: Any type
// CHECK-CC1-NEXT: Any value
// CHECK-CC1-NEXT: Enum tag
// CHECK-CC1-NEXT: Union tag
// CHECK-CC1-NEXT: Struct tag
// CHECK-CC1-NEXT: Class name
// CHECK-CC1-NEXT: Nested name specifier
// CHECK-CC1-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:11:11 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: OverloadCandidate:{ResultType void}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC2: Completion contexts:
// CHECK-CC2-NEXT: Any type
// CHECK-CC2-NEXT: Any value
// CHECK-CC2-NEXT: Enum tag
// CHECK-CC2-NEXT: Union tag
// CHECK-CC2-NEXT: Struct tag
// CHECK-CC2-NEXT: Class name
// CHECK-CC2-NEXT: Nested name specifier
// CHECK-CC2-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:12:18 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: OverloadCandidate:{ResultType void}{LeftParen (}{Placeholder int}{Optional {Comma , }{CurrentParameter ...}}{RightParen )} (1)
// CHECK-CC3: Completion contexts:
// CHECK-CC3-NEXT: Any type
// CHECK-CC3-NEXT: Any value
// CHECK-CC3-NEXT: Enum tag
// CHECK-CC3-NEXT: Union tag
// CHECK-CC3-NEXT: Struct tag
// CHECK-CC3-NEXT: Class name
// CHECK-CC3-NEXT: Nested name specifier
// CHECK-CC3-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:13:18 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: OverloadCandidate:{ResultType void}{LeftParen (}{Placeholder int}{Optional {Comma , }{CurrentParameter ...}}{RightParen )} (1)
// CHECK-CC4: Completion contexts:
// CHECK-CC4-NEXT: Any type
// CHECK-CC4-NEXT: Any value
// CHECK-CC4-NEXT: Enum tag
// CHECK-CC4-NEXT: Union tag
// CHECK-CC4-NEXT: Struct tag
// CHECK-CC4-NEXT: Class name
// CHECK-CC4-NEXT: Nested name specifier
// CHECK-CC4-NEXT: Objective-C interface
