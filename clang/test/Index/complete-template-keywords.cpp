templ
// RUN: env c-index-test -code-completion-at=%s:1:5 %s | FileCheck -check-prefix=CHECK-NO-PATTERN %s
// CHECK-NO-PATTERN: {TypedText template} (1)
// RUN: env CINDEXTEST_CODE_COMPLETE_PATTERNS=1 c-index-test -code-completion-at=%s:1:5 %s | FileCheck -check-prefix=CHECK-PATTERN %s
// CHECK-PATTERN: {TypedText template}{LeftAngle <}
