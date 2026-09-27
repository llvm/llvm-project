// REQUIRES: riscv-registered-target

/// Regression test for https://github.com/llvm/llvm-project/pull/213410:
/// Check that -march=rv64gcv -flto records +d in module asm and function
/// target-features even though the driver only passes mcpu=generic-rv64 to lld.

// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -flto %s -S -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefix=IR

// IR:      module asm(target_features: "{{.*}}+d{{.*}}", target_cpu: "generic-rv64")
// IR-NEXT: "nop"
// IR:      define dso_local void @_start() #[[#ATTR:]] {{.*}} {
// IR-NEXT: entry:
// IR-NEXT:   call void asm sideeffect "nop", ""()
// IR-NEXT:   ret void
// IR-NEXT: }
// IR-EMPTY:
// IR-NEXT: attributes #[[#ATTR]] = { {{.*}}"target-cpu"="generic-rv64" "target-features"="{{.*}}+d{{.*}}"
// IR:      ![[#]] = !{i32 1, !"target-abi", !"lp64d"}
// IR-NEXT: ![[#]] = !{i32 6, !"riscv-isa", ![[#ISA:]]}
// IR-NEXT: ![[#ISA]] = !{!"{{.*}}_d2p2_{{.*}}"}

// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -flto -shared -nostdlib -fuse-ld=lld %s -### 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER

// DRIVER: "-cc1"{{.*}}"-target-cpu" "generic-rv64"{{.*}}"-target-feature" "+d"{{.*}}"-target-abi" "lp64d"
// DRIVER: "{{[^"]*}}ld.lld{{(\.exe)?}}"
// DRIVER-NOT: mattr
// DRIVER-SAME: "-plugin-opt=mcpu=generic-rv64"
// DRIVER-NOT: mattr

__asm__("nop");

void _start(void) {
  __asm__ volatile("nop");
}
