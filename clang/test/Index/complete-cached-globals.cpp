// Note: the run lines follow their respective tests, since line/column
// matter in this test.

namespace SomeNamespace {
    class SomeClass {
    };
    void SomeFunction();
}

using SomeNamespace::SomeClass;
using SomeNamespace::SomeFunction;

static void foo() {
  return;
}

// rdar://23454249

// RUN: c-index-test -code-completion-at=%s:14:3 %s | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:14:3 %s | FileCheck -check-prefix=CHECK-CC1 %s

// CHECK-CC1: ClassDecl:{TypedText SomeClass} (50)
// CHECK-CC1: FunctionDecl:{ResultType void}{TypedText SomeFunction}{LeftParen (}{RightParen )} (50)
// CHECK-CC1-NOT: {Text SomeNamespace::}{TypedText SomeClass}
// CHECK-CC1-NOT: {Text SomeNamespace::}{TypedText SomeFunction}
