/// Test -m[no]div32 options.

// RUN: %clang --target=loongarch64 -mdiv32 -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-DIV32
// RUN: %clang --target=loongarch64 -mno-div32 -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-DIV32
// RUN: %clang --target=loongarch64 -mno-div32 -mdiv32 -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-DIV32
// RUN: %clang --target=loongarch64  -mdiv32 -mno-div32 -fsyntax-only %s -### 2>&1 | \
// RUN:     FileCheck %s --check-prefix=CC1-NO-DIV32

// RUN: %clang --target=loongarch64 -mdiv32 -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-DIV32
// RUN: %clang --target=loongarch64 -mno-div32 -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-DIV32
// RUN: %clang --target=loongarch64 -mno-div32 -mdiv32 -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-DIV32
// RUN: %clang --target=loongarch64 -mdiv32 -mno-div32 -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --check-prefix=IR-NO-DIV32


// CC1-DIV32: "-target-feature" "+div32"
// CC1-NO-DIV32: "-target-feature" "-div32"

// IR-DIV32: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+div32{{(,.*)?}}"
// IR-NO-DIV32: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-div32{{(,.*)?}}"

int foo(void) {
  return 42;
}
