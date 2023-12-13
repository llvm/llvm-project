/// Test -m[no-]lsx options.

// RUN: %clang --target=loongarch64 -mlsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LSX
// RUN: %clang --target=loongarch64 -mno-lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NOLSX
// RUN: %clang --target=loongarch64 -mlsx -mno-lsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NOLSX
// RUN: %clang --target=loongarch64 -mno-lsx -mlsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LSX
// RUN: %clang --target=loongarch64 -mlsx -mno-lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LSX
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LSX
// RUN: %clang --target=loongarch64 -mno-lsx -mno-lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NOLSX

// RUN: %clang --target=loongarch64 -mlsx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LSX
// RUN: %clang --target=loongarch64 -mno-lsx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NOLSX
// RUN: %clang --target=loongarch64 -mlsx -mno-lsx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NOLSX
// RUN: %clang --target=loongarch64 -mno-lsx -mlsx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LSX
// RUN: %clang --target=loongarch64 -mlsx -mno-lasx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LSX
// RUN: %clang --target=loongarch64 -mno-lasx -mlsx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LSX
// RUN: %clang --target=loongarch64 -mno-lsx -mno-lasx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NOLSX

// CC1-LSX: "-target-feature" "+lsx"
// CC1-NOLSX: "-target-feature" "-lsx"

// IR-LSX: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+lsx{{(,.*)?}}"
// IR-NOLSX: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-lsx{{(,.*)?}}"

int foo(void){
  return 3;
}
