/// Test -m[no]ld-seq-sa options.

// RUN: %clang --target=loongarch64 -mld-seq-sa -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-ld-seq-sa
// RUN: %clang --target=loongarch64 -mno-ld-seq-sa -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-ld-seq-sa
// RUN: %clang --target=loongarch64 -mno-ld-seq-sa -mld-seq-sa -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-ld-seq-sa
// RUN: %clang --target=loongarch64  -mld-seq-sa -mno-ld-seq-sa -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-ld-seq-sa

// RUN: %clang --target=loongarch64 -mld-seq-sa -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-ld-seq-sa
// RUN: %clang --target=loongarch64 -mno-ld-seq-sa -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-ld-seq-sa
// RUN: %clang --target=loongarch64 -mno-ld-seq-sa -mld-seq-sa -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-ld-seq-sa
// RUN: %clang --target=loongarch64 -mld-seq-sa -mno-ld-seq-sa -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-ld-seq-sa


// CC1-ld-seq-sa: "-target-feature" "+ld-seq-sa"
// CC1-NO-ld-seq-sa: "-target-feature" "-ld-seq-sa"

// IR-ld-seq-sa: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+ld-seq-sa{{(,.*)?}}"
// IR-NO-ld-seq-sa: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-ld-seq-sa{{(,.*)?}}"

int foo(void) {
  return 42;
}
