// RUN: %clang --target=loongarch64 -msoft-float -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1
// RUN: %clang --target=loongarch64 -msoft-float -mfpu=0 -mabi=lp64s -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,NOWARN
// RUN: %clang --target=loongarch64 -msoft-float -mfpu=64 -mabi=lp64d -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,WARN
// RUN: %clang --target=loongarch64 -msoft-float -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR

// NOWARN-NOT: warning:
// WARN: warning: ignoring '-mabi=lp64d' as it conflicts with that implied by '-msoft-float' (lp64s)
// WARN: warning: ignoring '-mfpu=64' as it conflicts with that implied by '-msoft-float' (0)

// CC1: "-target-feature" "-f"{{.*}} "-target-feature" "-d" "-target-feature" "-lsx"
// CC1: "-target-abi" "lp64s"

// IR: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}-d,{{(.*,)?}}-f,-lsx"

int foo(void) {
  return 3;
}
