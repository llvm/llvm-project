/// Test -m[no]scq options.

// RUN: %clang --target=loongarch64 -mscq -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-SCQ
// RUN: %clang --target=loongarch64 -mno-scq -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-SCQ
// RUN: %clang --target=loongarch64 -mno-scq -mscq -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-SCQ
// RUN: %clang --target=loongarch64  -mscq -mno-scq -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-SCQ

// RUN: %clang --target=loongarch64 -mscq -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-SCQ
// RUN: %clang --target=loongarch64 -mno-scq -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-SCQ
// RUN: %clang --target=loongarch64 -mno-scq -mscq -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-SCQ
// RUN: %clang --target=loongarch64 -mscq -mno-scq -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-SCQ


// CC1-SCQ: "-target-feature" "+scq"
// CC1-NO-SCQ: "-target-feature" "-scq"

// IR-SCQ: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+scq{{(,.*)?}}"
// IR-NO-SCQ: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-scq{{(,.*)?}}"

int foo(void) {
  return 42;
}