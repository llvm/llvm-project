// Test --sycl-device-link: SYCL device code is linked and wrapped into a host
// object, which is copied to the output. The host linker is not run.

// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple spirv64-unknown-unknown -emit-llvm-bc -o %t.spirv.bc

// RUN: llvm-offload-binary -o %t.sycl.out \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.sycl.o \
// RUN:   -fembed-offload-object=%t.sycl.out

// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not=/usr/bin/ld
// LINK: "{{.*}}clang{{(.exe)?}}" {{.*}}--target=spirv64-unknown-unknown {{.*}}.o --sycl-link{{$}}
// LINK-NEXT: "cp" {{.*}}dev.o.sycl.image.wrapper{{.*}}.o dev.o{{$}}

// Archive members are only extracted for the host link, so use
// --whole-archive to device link a static library.
// RUN: rm -f %t.sycl.a && llvm-ar rcs %t.sycl.a %t.sycl.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld --whole-archive %t.sycl.a \
// RUN:   --no-whole-archive -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not=/usr/bin/ld

// The output is the wrapped device object with SYCL runtime registration.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o -o %t.dev.o
// RUN: llvm-nm %t.dev.o | FileCheck %s --check-prefix=WRAPPED
// WRAPPED-DAG: .sycl_offloading.binary
// WRAPPED-DAG: U __sycl_register_lib

// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o -o dev.obj 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIN --implicit-check-not=/usr/bin/ld
// WIN: "copy" {{.*}}dev.obj.sycl.image.wrapper{{.*}}.o dev.obj{{$}}

// No device code: an empty host object without SYCL runtime registration is
// written to the output.
// RUN: rm -f %t.empty.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.elf.o -o %t.empty.o
// RUN: llvm-readobj --file-header %t.empty.o | FileCheck %s --check-prefix=EMPTY
// RUN: llvm-nm %t.empty.o | count 0
// EMPTY: Format: elf64-x86-64

// RUN: llvm-offload-binary -o %t.openmp.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.openmp.o \
// RUN:   -fembed-offload-object=%t.openmp.out
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.openmp.o -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOT-SYCL
// NOT-SYCL: error: --sycl-device-link requires SYCL device code, got 'openmp'

// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --emit-fatbin-only --linker-path=/usr/bin/ld \
// RUN:   %t.sycl.o -o dev.o 2>&1 | FileCheck %s --check-prefix=FATBIN
// FATBIN: error: --sycl-device-link cannot be used with --emit-fatbin-only
