// RUN: %clang --target=loongarch64 -msingle-float -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1
// RUN: %clang --target=loongarch64 -msingle-float -mfpu=32 -mabi=lp64f -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,NOWARN
// RUN: %clang --target=loongarch64 -msingle-float -mfpu=64 -mabi=lp64s -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,WARN
// RUN: %clang --target=loongarch64 -msingle-float -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR

// NOWARN-NOT: warning:
// WARN: warning: ignoring '-mabi=lp64s' as it conflicts with that implied by '-msingle-float' (lp64f)
// WARN: warning: ignoring '-mfpu=64' as it conflicts with that implied by '-msingle-float' (32)

// CC1: "-target-feature" "+f"{{.*}} "-target-feature" "-d" "-target-feature" "-lsx"
// CC1: "-target-abi" "lp64f"

// IR: attributes #[[#]] ={{.*}}"target-features"="{{(.*,)?}}+f,{{(.*,)?}}-d,-lsx"

int foo(void) {
  return 3;
}
