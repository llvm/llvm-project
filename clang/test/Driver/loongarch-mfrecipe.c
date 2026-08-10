/// Test -m[no]frecipe options.

// RUN: %clang --target=loongarch64 -mfrecipe -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-FRECIPE
// RUN: %clang --target=loongarch64 -mno-frecipe -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-FRECIPE
// RUN: %clang --target=loongarch64 -mno-frecipe -mfrecipe -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-FRECIPE
// RUN: %clang --target=loongarch64  -mfrecipe -mno-frecipe -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-FRECIPE

// RUN: %clang --target=loongarch64 -mfrecipe -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-FRECIPE
// RUN: %clang --target=loongarch64 -mno-frecipe -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-FRECIPE
// RUN: %clang --target=loongarch64 -mno-frecipe -mfrecipe -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-FRECIPE
// RUN: %clang --target=loongarch64 -mfrecipe -mno-frecipe -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-FRECIPE


// CC1-FRECIPE: "-target-feature" "+frecipe"
// CC1-NO-FRECIPE: "-target-feature" "-frecipe"

// IR-FRECIPE: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+frecipe{{(,.*)?}}"
// IR-NO-FRECIPE: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-frecipe{{(,.*)?}}"

int foo(void) {
  return 42;
}