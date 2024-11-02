// RUN: %clang --target=loongarch64 -mdouble-float -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1
// RUN: %clang --target=loongarch64 -mdouble-float -mfpu=0 -mabi=lp64s -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=CC1,WARN
// RUN: %clang --target=loongarch64 -mdouble-float -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR

// WARN: warning: argument unused during compilation: '-mfpu=0'
// WARN: warning: argument unused during compilation: '-mabi=lp64s'

// CC1-NOT: "-target-feature"
// CC1: "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "+d"
// CC1-NOT: "-target-feature"
// CC1: "-target-abi" "lp64d"

// IR: attributes #[[#]] ={{.*}}"target-features"="+64bit,+d,+f"

int foo(void) {
  return 3;
}
