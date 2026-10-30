// RUN: %clang -### -fexperimental-call-graph-section %s 2>&1 | FileCheck --check-prefix=CALL-GRAPH-SECTION %s
// RUN: %clang -### -fexperimental-call-graph-section -fno-experimental-call-graph-section %s 2>&1 | FileCheck --check-prefix=NO-CALL-GRAPH-SECTION %s

// CALL-GRAPH-SECTION: "-fexperimental-call-graph-section"
// NO-CALL-GRAPH-SECTION-NOT: "-fexperimental-call-graph-section"

// The call graph section type identifiers honor integer normalization like CFI
// does, so the option is forwarded even when no sanitizer is enabled.
// RUN: %clang -### -fexperimental-call-graph-section -fsanitize-cfi-icall-experimental-normalize-integers %s 2>&1 | FileCheck --check-prefix=NORMALIZE-INTEGERS %s
// RUN: %clang -### -fsanitize-cfi-icall-experimental-normalize-integers %s 2>&1 | FileCheck --check-prefix=NO-NORMALIZE-INTEGERS %s

// NORMALIZE-INTEGERS-NOT: warning: argument unused during compilation
// NORMALIZE-INTEGERS: "-fexperimental-call-graph-section"
// NORMALIZE-INTEGERS-SAME: "-fsanitize-cfi-icall-experimental-normalize-integers"
// NO-NORMALIZE-INTEGERS-NOT: "-fsanitize-cfi-icall-experimental-normalize-integers"
