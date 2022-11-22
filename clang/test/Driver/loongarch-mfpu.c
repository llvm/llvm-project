// RUN: %clang --target=loongarch64 -mfpu=64 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-FPU64
// RUN: %clang --target=loongarch64 -mfpu=32 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-FPU32
// RUN: %clang --target=loongarch64 -mfpu=0 -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-FPU0
// RUN: %clang --target=loongarch64 -mfpu=none -fsyntax-only %s -### 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CC1-FPU0

// RUN: %clang --target=loongarch64 -mfpu=64 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-FPU64
// RUN: %clang --target=loongarch64 -mfpu=32 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-FPU32
// RUN: %clang --target=loongarch64 -mfpu=0 -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-FPU0
// RUN: %clang --target=loongarch64 -mfpu=none -S -emit-llvm %s -o - | \
// RUN:   FileCheck %s --check-prefix=IR-FPU0

// CC1-FPU64-NOT: "-target-feature"
// CC1-FPU64: "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "+d"
// CC1-FPU64-NOT: "-target-feature"
// CC1-FPU64: "-target-abi" "lp64d"

// CC1-FPU32-NOT: "-target-feature"
// CC1-FPU32: "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "-d"
// CC1-FPU32-NOT: "-target-feature"
// CC1-FPU32: "-target-abi" "lp64f"

// CC1-FPU0-NOT: "-target-feature"
// CC1-FPU0: "-target-feature" "+64bit" "-target-feature" "-f" "-target-feature" "-d"
// CC1-FPU0-NOT: "-target-feature"
// CC1-FPU0: "-target-abi" "lp64s"

// IR-FPU64: attributes #[[#]] ={{.*}}"target-features"="+64bit,+d,+f"
// IR-FPU32: attributes #[[#]] ={{.*}}"target-features"="+64bit,+f,-d"
// IR-FPU0: attributes #[[#]] ={{.*}}"target-features"="+64bit,-d,-f"

int foo(void) {
  return 3;
}
