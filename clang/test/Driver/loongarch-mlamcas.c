/// Test -m[no]lamcas options.

// RUN: %clang --target=loongarch64 -mlamcas -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-LAMCAS
// RUN: %clang --target=loongarch64 -mno-lamcas -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-LAMCAS
// RUN: %clang --target=loongarch64 -mno-lamcas -mlamcas -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-LAMCAS
// RUN: %clang --target=loongarch64  -mlamcas -mno-lamcas -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-LAMCAS

// RUN: %clang --target=loongarch64 -mlamcas -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-LAMCAS
// RUN: %clang --target=loongarch64 -mno-lamcas -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-LAMCAS
// RUN: %clang --target=loongarch64 -mno-lamcas -mlamcas -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-LAMCAS
// RUN: %clang --target=loongarch64 -mlamcas -mno-lamcas -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-LAMCAS


// CC1-LAMCAS: "-target-feature" "+lamcas"
// CC1-NO-LAMCAS: "-target-feature" "-lamcas"

// IR-LAMCAS: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+lamcas{{(,.*)?}}"
// IR-NO-LAMCAS: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-lamcas{{(,.*)?}}"

int foo(void) {
  return 42;
}
