/// Test -m[no-]unaligned-access and -m[no-]strict-align options.

// RUN: %clang --target=loongarch64 -munaligned-access -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-unaligned-access -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -munaligned-access -mno-unaligned-access -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-unaligned-access -munaligned-access -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -mno-strict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -mstrict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -munaligned-access -mstrict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -munaligned-access -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-unaligned-access -mno-strict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -mno-unaligned-access -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED

// RUN: %clang --target=loongarch64 -munaligned-access -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-unaligned-access -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -munaligned-access -mno-unaligned-access -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-unaligned-access -munaligned-access -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -mno-strict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -mstrict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -munaligned-access -mstrict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -munaligned-access -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-unaligned-access -mno-strict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -mno-unaligned-access -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED

// CC1-UNALIGNED: "-target-feature" "+ual"
// CC1-NO-UNALIGNED: "-target-feature" "-ual"

// IR-UNALIGNED: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+ual{{(,.*)?}}"
// IR-NO-UNALIGNED: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-ual{{(,.*)?}}"

int foo(void) {
  return 3;
}
