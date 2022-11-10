// RUN: %clang --target=loongarch64 -mfpu=64 -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-FPU64
// RUN: %clang --target=loongarch64 -mfpu=32 -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-FPU32
// RUN: %clang --target=loongarch64 -mfpu=0 -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-FPU0
// RUN: %clang --target=loongarch64 -mfpu=none -fsyntax-only %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CC1-FPU0

// RUN: %clang --target=loongarch64 -mfpu=64 -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=IR-FPU64
// RUN: %clang --target=loongarch64 -mfpu=32 -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=IR-FPU32
// RUN: %clang --target=loongarch64 -mfpu=0 -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=IR-FPU0
// RUN: %clang --target=loongarch64 -mfpu=none -S -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=IR-FPU0

// CC1-FPU64-NOT: "-target-feature"
// CC1-FPU64: "-target-feature" "+64bit"
// CC1-FPU64-SAME: "-target-feature" "+f"
// CC1-FPU64-SAME: "-target-feature" "+d"
// CC1-FPU64-NOT: "-target-feature"
// CC1-FPU64: "-target-abi" "lp64d"

// CC1-FPU32-NOT: "-target-feature"
// CC1-FPU32: "-target-feature" "+64bit"
// CC1-FPU32-SAME: "-target-feature" "+f"
// CC1-FPU32-SAME: "-target-feature" "-d"
// CC1-FPU32-NOT: "-target-feature"
// CC1-FPU32: "-target-abi" "lp64f"

// CC1-FPU0-NOT: "-target-feature"
// CC1-FPU0: "-target-feature" "+64bit"
// CC1-FPU0-SAME: "-target-feature" "-f"
// CC1-FPU0-SAME: "-target-feature" "-d"
// CC1-FPU0-NOT: "-target-feature"
// CC1-FPU0: "-target-abi" "lp64s"

// IR-FPU64: attributes #{{[0-9]+}} ={{.*}}"target-features"="+64bit,+d,+f"
// IR-FPU32: attributes #{{[0-9]+}} ={{.*}}"target-features"="+64bit,+f,-d"
// IR-FPU0: attributes #{{[0-9]+}} ={{.*}}"target-features"="+64bit,-d,-f"

/// Dummy function
int foo(void) {
  return  3;
}
