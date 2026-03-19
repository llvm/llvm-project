/// Test -m[no-]annotate-tablejump options.

// RUN: %clang --target=loongarch64 -mannotate-tablejump %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-ANOTATE
// RUN: %clang --target=loongarch64 -mno-annotate-tablejump %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-ANOTATE
// RUN: %clang --target=loongarch64 -mannotate-tablejump -mno-annotate-tablejump %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-ANOTATE
// RUN: %clang --target=loongarch64 -mno-annotate-tablejump -mannotate-tablejump %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-ANOTATE

// CC1-ANOTATE: "-loongarch-annotate-tablejump"
// CC1-NO-ANOTATE-NOT: "-loongarch-annotate-tablejump"
