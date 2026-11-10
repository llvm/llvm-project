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
// LINK: "{{.*}}clang{{(.exe)?}}" {{.*}}--target=spirv64-unknown-unknown {{.*}}sycl-spirv64-unknown-unknown-generic{{.*}}.o --sycl-link{{$}}

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
// RUN: llvm-readobj --sections %t.dev.o | FileCheck %s --check-prefix=SECTIONS
// WRAPPED-DAG: {{^[0-9a-f]+}} r .sycl_offloading.binary{{$}}
// WRAPPED-DAG: {{^ +}}U __sycl_register_lib{{$}}
// SECTIONS-NOT: Name: .llvm.offloading
// SECTIONS: Name: .sycl_fatbin
// SECTIONS-NOT: Name: .llvm.offloading

// The device image is already linked, so a later wrapped host link does not
// link it again.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.dev.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RELINK --implicit-check-not=--sycl-link
// RELINK: "/usr/bin/ld" {{.*}}dev.o -o a.out

// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc \
// RUN:   --sycl-device-link --linker-path=link.exe %t.sycl.o -o %t.dev.obj
// RUN: llvm-readobj --file-header %t.dev.obj | FileCheck %s --check-prefix=WIN
// WIN: Format: COFF-x86-64

// With -v and --save-temps, the wrapping is still done in-process, because the
// image must be embedded as finalized.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run -v \
// RUN:   --save-temps --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o \
// RUN:   -o dev.o 2>&1 | FileCheck %s --check-prefix=LINK \
// RUN:   --implicit-check-not=/usr/bin/ld --implicit-check-not=llvm-offload-wrapper

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
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --override-image=openmp=%t.elf.o \
// RUN:   --linker-path=/usr/bin/ld %t.elf.o -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOT-SYCL
// NOT-SYCL: error: --sycl-device-link requires SYCL device code, got 'openmp'

// Without a host link, a missing input must be diagnosed here.
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o %t.missing.o \
// RUN:   -o dev.o 2>&1 | FileCheck %s --check-prefix=MISSING -DFILE=%t.missing.o
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld --whole-archive \
// RUN:   %t.missing.o --no-whole-archive -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISSING -DFILE=%t.missing.o
// MISSING: error: '[[FILE]]': {{[Nn]}}o such file or directory

// A value of a host linker option is not reported as a missing input.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o -z now \
// RUN:   -Map out.map -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not=/usr/bin/ld

// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --emit-fatbin-only --linker-path=/usr/bin/ld \
// RUN:   %t.sycl.o -o dev.o 2>&1 | FileCheck %s --check-prefix=FATBIN
// FATBIN: error: --sycl-device-link cannot be used with --emit-fatbin-only
