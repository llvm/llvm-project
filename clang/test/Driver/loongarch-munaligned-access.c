/// Test -m[no-]unaligned-access and -m[no-]strict-align options.

// RUN: %clang --target=loongarch64 -mstrict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -mno-strict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -mstrict-align -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NO-UNALIGNED

// RUN: %clang --target=loongarch64 -mstrict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -mstrict-align -mno-strict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-UNALIGNED
// RUN: %clang --target=loongarch64 -mno-strict-align -mstrict-align -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NO-UNALIGNED

// RUN: not %clang -### --target=loongarch64 -mno-unaligned-access -munaligned-access %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR

// CC1-UNALIGNED: "-target-feature" "+ual"
// CC1-NO-UNALIGNED: "-target-feature" "-ual"

// IR-UNALIGNED: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+ual{{(,.*)?}}"
// IR-NO-UNALIGNED: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-ual{{(,.*)?}}"

// ERR: error: unsupported option '-mno-unaligned-access' for target 'loongarch64'
// ERR: error: unsupported option '-munaligned-access' for target 'loongarch64'

int foo(void) {
  return 3;
}
