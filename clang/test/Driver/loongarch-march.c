// RUN: %clang --target=loongarch64 -march=loongarch64 -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-LOONGARCH64
// RUN: %clang --target=loongarch64 -march=la464 -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-LA464
// RUN: %clang --target=loongarch64 -march=loongarch64 -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=IR-LOONGARCH64
// RUN: %clang --target=loongarch64 -march=la464 -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=IR-LA464

// CC1-LOONGARCH64-NOT: "-target-feature"
// CC1-LOONGARCH64: "-target-feature" "+64bit"
// CC1-LOONGARCH64-SAME: {{^}} "-target-feature" "+f"
// CC1-LOONGARCH64-SAME: {{^}} "-target-feature" "+d"
// CC1-LOONGARCH64-NOT: "-target-feature"
// CC1-LOONGARCH64: "-target-abi" "lp64d"

// CC1-LA464-NOT: "-target-feature"
// CC1-LA464: "-target-feature" "+64bit"
// CC1-LA464-SAME: {{^}} "-target-feature" "+f"
// CC1-LA464-SAME: {{^}} "-target-feature" "+d"
// CC1-LA464-SAME: {{^}} "-target-feature" "+lsx"
// CC1-LA464-SAME: {{^}} "-target-feature" "+lasx"
// CC1-LA464-NOT: "-target-feature"
// CC1-LA464: "-target-abi" "lp64d"

// IR-LOONGARCH64: attributes #{{[0-9]+}} ={{.*}}"target-features"="+64bit,+d,+f"
// IR-LA464: attributes #{{[0-9]+}} ={{.*}}"target-features"="+64bit,+d,+f,+lasx,+lsx"

/// Dummy function
int foo(void) {
  return  3;
}
