// Note: the run lines follow their respective tests, since line/column
// matter in this test.

void foo(int a = 42, int = 42);
void bar(int a, int b = 42, int c = 42);
void baz(int a = 42, ...);
struct S{ S(int a = 42, int = 42) {} };

class Bar1 { public: Bar1() {} }; class Bar2;
void foo_2(Bar1 b1 = Bar1(), Bar2 b2 = Bar2());

int main() {
    foo(42, 42);
    bar(42, 42, 42);
    baz(42, 42, 42);
    S s(42, 42);
    foo_2();
}

// RUN: c-index-test -code-completion-at=%s:13:9 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: OverloadCandidate:{ResultType void}{Text foo}{LeftParen (}{Optional {CurrentParameter int a = 42}{Optional {Comma , }{Placeholder int = 42}}}{RightParen )} (1)
// CHECK-CC1: Completion contexts:
// CHECK-CC1-NEXT: Any type
// CHECK-CC1-NEXT: Any value
// CHECK-CC1-NEXT: Enum tag
// CHECK-CC1-NEXT: Union tag
// CHECK-CC1-NEXT: Struct tag
// CHECK-CC1-NEXT: Class name
// CHECK-CC1-NEXT: Nested name specifier
// CHECK-CC1-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:14:9 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: OverloadCandidate:{ResultType void}{Text bar}{LeftParen (}{CurrentParameter int a}{Optional {Comma , }{Placeholder int b = 42}{Optional {Comma , }{Placeholder int c = 42}}}{RightParen )} (1)
// CHECK-CC2: Completion contexts:
// CHECK-CC2-NEXT: Any type
// CHECK-CC2-NEXT: Any value
// CHECK-CC2-NEXT: Enum tag
// CHECK-CC2-NEXT: Union tag
// CHECK-CC2-NEXT: Struct tag
// CHECK-CC2-NEXT: Class name
// CHECK-CC2-NEXT: Nested name specifier
// CHECK-CC2-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:14:16 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: OverloadCandidate:{ResultType void}{Text bar}{LeftParen (}{Placeholder int a}{Optional {Comma , }{Placeholder int b = 42}{Optional {Comma , }{CurrentParameter int c = 42}}}{RightParen )} (1)
// CHECK-CC3: Completion contexts:
// CHECK-CC3-NEXT: Any type
// CHECK-CC3-NEXT: Any value
// CHECK-CC3-NEXT: Enum tag
// CHECK-CC3-NEXT: Union tag
// CHECK-CC3-NEXT: Struct tag
// CHECK-CC3-NEXT: Class name
// CHECK-CC3-NEXT: Nested name specifier
// CHECK-CC3-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:15:16 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: OverloadCandidate:{ResultType void}{Text baz}{LeftParen (}{Optional {Placeholder int a = 42}{Optional {Comma , }{CurrentParameter ...}}}{RightParen )} (1)
// CHECK-CC4: Completion contexts:
// CHECK-CC4-NEXT: Any type
// CHECK-CC4-NEXT: Any value
// CHECK-CC4-NEXT: Enum tag
// CHECK-CC4-NEXT: Union tag
// CHECK-CC4-NEXT: Struct tag
// CHECK-CC4-NEXT: Class name
// CHECK-CC4-NEXT: Nested name specifier
// CHECK-CC4-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:16:9 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: OverloadCandidate:{Text S}{LeftParen (}{Optional {CurrentParameter int a = 42}{Optional {Comma , }{Placeholder int = 42}}}{RightParen )} (1)
// CHECK-CC5: OverloadCandidate:{Text S}{LeftParen (}{CurrentParameter const S &}{RightParen )} (1)
// CHECK-CC5: Completion contexts:
// CHECK-CC5-NEXT: Any type
// CHECK-CC5-NEXT: Any value
// CHECK-CC5-NEXT: Enum tag
// CHECK-CC5-NEXT: Union tag
// CHECK-CC5-NEXT: Struct tag
// CHECK-CC5-NEXT: Class name
// CHECK-CC5-NEXT: Nested name specifier
// CHECK-CC5-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:17:11 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: FunctionDecl:{ResultType void}{TypedText foo_2}{LeftParen (}{Optional {Placeholder Bar1 b1 = Bar1()}{Optional {Comma , }{Placeholder Bar2 b2 = Bar2()}}}{RightParen )} (50)
// CHECK-CC6: Completion contexts:
// CHECK-CC6-NEXT: Any type
// CHECK-CC6-NEXT: Any value
// CHECK-CC6-NEXT: Enum tag
// CHECK-CC6-NEXT: Union tag
// CHECK-CC6-NEXT: Struct tag
// CHECK-CC6-NEXT: Class name
// CHECK-CC6-NEXT: Nested name specifier
// CHECK-CC6-NEXT: Objective-C interface
