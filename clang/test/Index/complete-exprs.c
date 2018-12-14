// Note: the run lines follow their respective tests, since line/column
// matter in this test.

int f(int) __attribute__((unavailable));

int test(int i, int j, int k, int l) {
  return i | j | k & l;
}

struct X __attribute__((deprecated)) f1 = { 17 };
void f2() { f1(17); }

const char *str = "Hello, \nWorld";

void f3(const char*, ...) __attribute__((sentinel(0)));

#define NULL __null
void f4(const char* str) {
  f3(str, NULL);
}

typedef int type;
void f5(float f) {
  (type)f;
}

// RUN: c-index-test -code-completion-at=%s:7:10 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:7:10 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{TypedText __PRETTY_FUNCTION__} (65)
// CHECK-CC1: macro definition:{TypedText __VERSION__} (70)
// CHECK-CC1: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (12) (unavailable)
// CHECK-CC1-NOT: NotImplemented:{TypedText float} (65)
// CHECK-CC1: ParmDecl:{ResultType int}{TypedText j} (8)
// CHECK-CC1: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:7:10 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:7:14 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:7:14 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s

// RUN: c-index-test -code-completion-at=%s:7:18 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:7:22 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: c-index-test -code-completion-at=%s:7:2 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: macro definition:{TypedText __VERSION__} (70)
// CHECK-CC2: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC2: NotImplemented:{TypedText float} (50)
// CHECK-CC2: ParmDecl:{ResultType int}{TypedText j} (34)
// CHECK-CC2: NotImplemented:{ResultType size_t}{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: c-index-test -code-completion-at=%s:11:16 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: FunctionDecl:{ResultType int}{TypedText f}{LeftParen (}{Placeholder int}{RightParen )} (50)
// CHECK-CC4: VarDecl:{ResultType struct X}{TypedText f1} (50) (deprecated)

// RUN: c-index-test -code-completion-at=%s:19:3 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: FunctionDecl:{ResultType void}{TypedText f3}{LeftParen (}{Placeholder const char *, ...}{Text , NULL}{RightParen )} (50)
// CHECK-CC6: NotImplemented:{TypedText void} (50)
// CHECK-CC6: NotImplemented:{TypedText volatile} (50)

// RUN: c-index-test -code-completion-at=%s:24:4 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC7 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:24:4 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: ParmDecl:{ResultType float}{TypedText f} (34)
// CHECK-CC7: VarDecl:{ResultType struct X}{TypedText f1} (50) (deprecated)
// CHECK-CC7: FunctionDecl:{ResultType void}{TypedText f2}{LeftParen (}{RightParen )} (50)
// CHECK-CC7: FunctionDecl:{ResultType void}{TypedText f3}{LeftParen (}{Placeholder const char *, ...}{Text , NULL}{RightParen )} (50)
// CHECK-CC7: FunctionDecl:{ResultType void}{TypedText f4}{LeftParen (}{Placeholder const char *str}{RightParen )} (50)
// CHECK-CC7: FunctionDecl:{ResultType void}{TypedText f5}{LeftParen (}{Placeholder float f}{RightParen )} (50)
// CHECK-CC7: TypedefDecl:{TypedText type} (50)
