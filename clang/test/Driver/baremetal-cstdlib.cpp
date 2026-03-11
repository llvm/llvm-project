// UNSUPPORTED: system-windows

// RUN: rm -rf %t.dir/baremetal_cstdlib
// RUN: mkdir -p %t.dir/baremetal_cstdlib/bin
// RUN: mkdir -p %t.dir/baremetal_cstdlib/lib/clang-runtimes
// RUN: mkdir -p %t.dir/baremetal_cstdlib/lib/clang-runtimes/newlib
// RUN: mkdir -p %t.dir/baremetal_cstdlib/lib/clang-runtimes/newlib-nano
// RUN: mkdir -p %t.dir/baremetal_cstdlib/lib/clang-runtimes/picolibc
// RUN: mkdir -p %t.dir/baremetal_cstdlib/lib/clang-runtimes/llvm-libc
// RUN: cp %S/baremetal-multilib.yaml %t.dir/baremetal_cstdlib/lib/clang-runtimes/multilib.yaml
// RUN: cp %S/baremetal-multilib.yaml %t.dir/baremetal_cstdlib/lib/clang-runtimes/newlib/multilib.yaml
// RUN: cp %S/baremetal-multilib.yaml %t.dir/baremetal_cstdlib/lib/clang-runtimes/newlib-nano/multilib.yaml
// RUN: cp %S/baremetal-multilib.yaml %t.dir/baremetal_cstdlib/lib/clang-runtimes/picolibc/multilib.yaml
// RUN: cp %S/baremetal-multilib.yaml %t.dir/baremetal_cstdlib/lib/clang-runtimes/llvm-libc/multilib.yaml
// RUN: ln -s %clang %t.dir/baremetal_cstdlib/bin/clang

// RUN: %t.dir/baremetal_cstdlib/bin/clang -no-canonical-prefixes %s -### -o %t.out 2>&1 \
// RUN:     -target thumbv8m.main-none-eabihf -mfpu=fpv5-d16 \
// RUN:   | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT: "-internal-isystem" "{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-DEFAULT: "-L{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}lib"

// RUN: %t.dir/baremetal_cstdlib/bin/clang -no-canonical-prefixes %s -### -o %t.out 2>&1 \
// RUN:     -target thumbv8m.main-none-eabihf -mfpu=fpv5-d16 --cstdlib=system \
// RUN:   | FileCheck --check-prefix=CHECK-SYSTEM %s
// CHECK-SYSTEM: "-internal-isystem" "{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-SYSTEM: "-L{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}lib"

// RUN: %t.dir/baremetal_cstdlib/bin/clang -no-canonical-prefixes %s -### -o %t.out 2>&1 \
// RUN:     -target thumbv8m.main-none-eabihf -mfpu=fpv5-d16 --cstdlib=picolibc \
// RUN:   | FileCheck --check-prefix=CHECK-PICOLIBC %s
// CHECK-PICOLIBC: "-internal-isystem" "{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}picolibc{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-PICOLIBC: "-L{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}picolibc{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}lib"

// RUN: %t.dir/baremetal_cstdlib/bin/clang -no-canonical-prefixes %s -### -o %t.out 2>&1 \
// RUN:     -target thumbv8m.main-none-eabihf -mfpu=fpv5-d16 --cstdlib=newlib \
// RUN:   | FileCheck --check-prefix=CHECK-NEWLIB %s
// CHECK-NEWLIB: "-internal-isystem" "{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}newlib{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-NEWLIB: "-L{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}newlib{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}lib"

// RUN: %t.dir/baremetal_cstdlib/bin/clang -no-canonical-prefixes %s -### -o %t.out 2>&1 \
// RUN:     -target thumbv8m.main-none-eabihf -mfpu=fpv5-d16 --cstdlib=newlib-nano \
// RUN:   | FileCheck --check-prefix=CHECK-NEWLIB-NANO %s
// CHECK-NEWLIB-NANO: "-internal-isystem" "{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}newlib-nano{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-NEWLIB-NANO: "-L{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}newlib-nano{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}lib"

// RUN: %t.dir/baremetal_cstdlib/bin/clang -no-canonical-prefixes %s -### -o %t.out 2>&1 \
// RUN:     -target thumbv8m.main-none-eabihf -mfpu=fpv5-d16 --cstdlib=llvm-libc \
// RUN:   | FileCheck --check-prefix=CHECK-LLVM-LIBC %s
// CHECK-LLVM-LIBC: "-internal-isystem" "{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}llvm-libc{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}include{{[/\\]+}}c++{{[/\\]+}}v1"
// CHECK-LLVM-LIBC: "-L{{.*}}baremetal_cstdlib{{[/\\]+}}bin{{[/\\]+}}..{{[/\\]+}}lib{{[/\\]+}}clang-runtimes{{[/\\]+}}llvm-libc{{[/\\]+}}arm-none-eabi{{[/\\]+}}thumb{{[/\\]+}}v8-m.main{{[/\\]+}}fp{{[/\\]+}}lib"

int main() { return 0; }
