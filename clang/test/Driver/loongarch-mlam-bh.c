/// Test -m[no]lam-bh options.

// RUN: %clang --target=loongarch64 -mlam-bh -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-LAM-BH
// RUN: %clang --target=loongarch64 -mno-lam-bh -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-LAM-BH
// RUN: %clang --target=loongarch64 -mno-lam-bh -mlam-bh -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-LAM-BH
// RUN: %clang --target=loongarch64  -mlam-bh -mno-lam-bh -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-LAM-BH

// RUN: %clang --target=loongarch64 -mlam-bh -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-LAM-BH
// RUN: %clang --target=loongarch64 -mno-lam-bh -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-LAM-BH
// RUN: %clang --target=loongarch64 -mno-lam-bh -mlam-bh -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-LAM-BH
// RUN: %clang --target=loongarch64 -mlam-bh -mno-lam-bh -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-LAM-BH


// CC1-LAM-BH: "-target-feature" "+lam-bh"
// CC1-NO-LAM-BH: "-target-feature" "-lam-bh"

// IR-LAM-BH: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+lam-bh{{(,.*)?}}"
// IR-NO-LAM-BH: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-lam-bh{{(,.*)?}}"

int foo(void) {
  return 42;
}