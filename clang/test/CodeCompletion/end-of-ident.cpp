class classifier {};
// We should get all three completions when the cursor is at the beginning,
// middle, or end.
class cls
// ^class cls
// RUN: %clang_cc1 -code-completion-at=%s:%(line-2):1 %s | FileCheck --check-prefix=CHECK-CLS %s
// cl^ass cls
// RUN: %clang_cc1 -code-completion-at=%s:%(line-4):3 %s | FileCheck --check-prefix=CHECK-CLS %s
// class^ cls
// RUN: %clang_cc1 -code-completion-at=%s:%(line-6):6 %s | FileCheck --check-prefix=CHECK-CLS %s

// CHECK-CLS: COMPLETION: class{{$}}
// CHECK-CLS: COMPLETION: classifier : classifier

// class ^cls
// RUN: %clang_cc1 -code-completion-at=%s:%(line-12):7 %s | FileCheck --check-prefix=CHECK-NO-CLS %s
// class c^ls
// RUN: %clang_cc1 -code-completion-at=%s:%(line-14):8 %s | FileCheck --check-prefix=CHECK-NO-CLS %s
// CHECK-NO-CLS-NOT: COMPLETION: class{{$}}
// CHECK-NO-CLS: COMPLETION: classifier : classifier
