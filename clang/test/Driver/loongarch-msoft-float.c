// RUN: %clang --target=loongarch64 -msoft-float -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1
// RUN: %clang --target=loongarch64 -msoft-float -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=IR

// CC1-NOT: "-target-feature"
// CC1: "-target-feature" "+64bit"
// CC1-SAME: {{^}} "-target-feature" "-f"
// CC1-SAME: {{^}} "-target-feature" "-d"
// CC1-NOT: "-target-feature"
// CC1: "-target-abi" "lp64s"

// IR: attributes #{{[0-9]+}} ={{.*}}"target-features"="+64bit,-d,-f"

/// Dummy function
int foo(void) {
  return  3;
}
