// RUN: %clang --target=loongarch64 -mdouble-float -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1
// RUN: %clang --target=loongarch64 -mdouble-float -mfpu=64 -mabi=lp64d -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,NOWARN
// RUN: %clang --target=loongarch64 -mdouble-float -mfpu=0 -mabi=lp64s -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,WARN,WARN-FPU0
// RUN: %clang --target=loongarch64 -mdouble-float -mfpu=none -mabi=lp64s -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,WARN,WARN-FPUNONE
// RUN: %clang --target=loongarch64 -mdouble-float -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR

// NOWARN-NOT: warning:
// WARN: warning: ignoring '-mabi=lp64s' as it conflicts with that implied by '-mdouble-float' (lp64d)
// WARN-FPU0: warning: ignoring '-mfpu=0' as it conflicts with that implied by '-mdouble-float' (64)
// WARN-FPUNONE: warning: ignoring '-mfpu=none' as it conflicts with that implied by '-mdouble-float' (64)

// CC1: "-target-feature" "+f"{{.*}} "-target-feature" "+d"
// CC1: "-target-abi" "lp64d"

// IR: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+d,{{(.*,)?}}+f{{(,.*)?}}"

int foo(void) {
  return 3;
}
