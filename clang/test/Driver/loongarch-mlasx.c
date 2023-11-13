/// Test -m[no-]lasx options.

// RUN: %clang --target=loongarch64 -mlasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LASX
// RUN: %clang --target=loongarch64 -mno-lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mlasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LASX
// RUN: %clang --target=loongarch64 -mlsx -mlasx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LASX
// RUN: %clang --target=loongarch64 -mlasx -mlsx -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-LASX

// RUN: %clang --target=loongarch64 -mlasx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LASX
// RUN: %clang --target=loongarch64 -mno-lasx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NOLASX
// RUN: %clang --target=loongarch64 -mlasx -mno-lasx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-NOLASX
// RUN: %clang --target=loongarch64 -mno-lasx -mlasx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LASX
// RUN: %clang --target=loongarch64 -mlsx -mlasx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LASX
// RUN: %clang --target=loongarch64 -mlasx -mlsx -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-LASX

// CC1-LASX: "-target-feature" "+lsx" "-target-feature" "+lasx"
// CC1-NOLASX: "-target-feature" "-lasx"

// IR-LASX: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+lasx{{(,.*)?}}"
// IR-NOLASX: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-lasx{{(,.*)?}}"

int foo(void){
  return 3;
}
