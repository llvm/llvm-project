#define FUNC(X) X
#define FUNCTOR
using FUNCTION = int();
// We should get all three completions when the cursor is at the beginning,
// middle, or end.
FUNC(int) a = 10;
// ^FUNC(int)
// RUN: %clang_cc1 -code-completion-at=%s:6:1 -code-completion-macros %s | FileCheck %s
// FU^NC(int)
// RUN: %clang_cc1 -code-completion-at=%s:6:3 -code-completion-macros %s | FileCheck %s
// FUNC^(int)
// RUN: %clang_cc1 -code-completion-at=%s:6:5 -code-completion-macros %s | FileCheck %s

// CHECK: COMPLETION: FUNC : FUNC(<#X#>)
// CHECK: COMPLETION: FUNCTION : FUNCTION
// CHECK: COMPLETION: FUNCTOR : FUNCTOR