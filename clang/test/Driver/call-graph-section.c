// RUN: %clang -### -S -fcall-graph-section %s 2>&1 | FileCheck --check-prefix=CALL-GRAPH-SECTION %s
// RUN: %clang -### -S -fcall-graph-section -fno-call-graph-section %s 2>&1 | FileCheck --check-prefix=NO-CALL-GRAPH-SECTION %s

// CALL-GRAPH-SECTION: "-fcall-graph-section"
// NO-CALL-GRAPH-SECTION-NOT: "-fcall-graph-section"
