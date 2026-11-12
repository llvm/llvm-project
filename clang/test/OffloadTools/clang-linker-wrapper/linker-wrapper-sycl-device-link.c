// Test --sycl-device-link: SYCL device code is linked and wrapped into a host
// object, which is copied to the output. The host linker is not run.

// REQUIRES: x86-registered-target
// REQUIRES: spirv-registered-target

// Inputs: a host object without device code (%t.elf.o), and host objects with
// SYCL device code for Linux (%t.sycl.o) and Windows (%t.win.obj).
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple spirv64-unknown-unknown -emit-llvm-bc -o %t.spirv.bc
// RUN: llvm-offload-binary -o %t.sycl.out \
// RUN:   --image=file=%t.spirv.bc,kind=sycl,triple=spirv64-unknown-unknown,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.sycl.o \
// RUN:   -fembed-offload-object=%t.sycl.out
// RUN: %clang -cc1 %s -triple x86_64-pc-windows-msvc -emit-obj -o %t.win.obj \
// RUN:   -fembed-offload-object=%t.sycl.out

//===----------------------------------------------------------------------===//
// The device code is linked, and the host linker is not run.
//===----------------------------------------------------------------------===//

// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not=/usr/bin/ld
// LINK: "{{.*}}clang{{(.exe)?}}" {{.*}}--target=spirv64-unknown-unknown {{.*}}sycl-spirv64-unknown-unknown-generic{{.*}}.o --sycl-link{{$}}

// 'now' in '-z now' and 'out.map' in '-Map out.map' are values of host linker
// options, not missing inputs.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o -z now \
// RUN:   -Map out.map -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not=/usr/bin/ld

// Device code in a static library is only extracted for device targets that
// the other inputs have. No other input has device code here, so
// --whole-archive is needed.
// RUN: rm -f %t.sycl.a && llvm-ar rcs %t.sycl.a %t.sycl.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld --whole-archive %t.sycl.a \
// RUN:   --no-whole-archive -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK --implicit-check-not=/usr/bin/ld

// With --wrapper-verbose and --save-temps, llvm-offload-wrapper is still not
// used, because it cannot mark images as finalized.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --wrapper-verbose --save-temps --sycl-device-link \
// RUN:   --linker-path=/usr/bin/ld %t.sycl.o -o dev.o 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LINK \
// RUN:   --implicit-check-not=/usr/bin/ld --implicit-check-not=llvm-offload-wrapper

//===----------------------------------------------------------------------===//
// The output file.
//===----------------------------------------------------------------------===//

// The output registers the device image with the SYCL runtime. The image is in
// .sycl_fatbin, not .llvm.offloading, because it is already linked.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o -o %t.dev.o
// RUN: llvm-nm %t.dev.o | FileCheck %s --check-prefix=WRAPPED
// RUN: llvm-readobj --sections %t.dev.o | FileCheck %s --check-prefix=SECTIONS
// WRAPPED-DAG: {{^[0-9a-f]+}} r .sycl_offloading.binary{{$}}
// WRAPPED-DAG: {{^ +}}U __sycl_register_lib{{$}}
// SECTIONS-NOT: Name: .llvm.offloading
// SECTIONS: Name: .sycl_fatbin
// SECTIONS-NOT: Name: .llvm.offloading

// Because the image is finalized, a later host link through
// clang-linker-wrapper does not link it again.
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld %t.dev.o -o a.out 2>&1 \
// RUN:   | FileCheck %s --check-prefix=RELINK --implicit-check-not=--sycl-link
// RELINK: "/usr/bin/ld" {{.*}}dev.o -o a.out

// Without device code, the output is an empty host object without SYCL runtime
// registration.
// RUN: rm -f %t.empty.o
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.elf.o -o %t.empty.o
// RUN: llvm-readobj --file-header %t.empty.o | FileCheck %s --check-prefix=EMPTY
// RUN: llvm-nm %t.empty.o | count 0
// EMPTY: Format: elf64-x86-64

//===----------------------------------------------------------------------===//
// Errors.
//===----------------------------------------------------------------------===//

// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --emit-fatbin-only --linker-path=/usr/bin/ld \
// RUN:   %t.sycl.o -o dev.o 2>&1 | FileCheck %s --check-prefix=FATBIN
// FATBIN: error: --sycl-device-link cannot be used with --emit-fatbin-only

// OpenMP device code is an error, both in an input and given with
// --override-image.
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

// No host linker runs, so clang-linker-wrapper reports missing inputs: a plain
// input, and one after '--'.
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld %t.sycl.o %t.missing.o \
// RUN:   -o dev.o 2>&1 | FileCheck %s --check-prefix=MISSING -DFILE=%t.missing.o
// RUN: not clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --sycl-device-link --linker-path=/usr/bin/ld -o dev.o -- %t.missing.o \
// RUN:   2>&1 | FileCheck %s --check-prefix=MISSING -DFILE=%t.missing.o
// MISSING: error: '[[FILE]]': {{[Nn]}}o such file or directory

//===----------------------------------------------------------------------===//
// Windows hosts: link.exe options are case-insensitive, and link.exe searches
// the LIB environment variable.
//===----------------------------------------------------------------------===//

// link.exe options such as /nologo and /LIBPATH:C:\dir are not reported as
// missing inputs. The -o names the output, because it comes after /OUT: and
// the last output option wins.
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --sycl-device-link --linker-path=link.exe %t.sycl.o /nologo \
// RUN:   /IGNORE:4078 '/LIBPATH:C:\dir' /OUT:C:/dir/a.exe -o dev.obj 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIN-NAME -DNAME=dev.obj

// /out: names the output in any case, and the last of -o and /out: wins, e.g.
// for clang-cl '/link /out:'.
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --sycl-device-link --linker-path=link.exe -o first.exe %t.sycl.o \
// RUN:   /Out:last.obj 2>&1 | FileCheck %s --check-prefix=WIN-NAME -DNAME=last.obj
// WIN-NAME: -dumpdir [[NAME]].spirv64.

// Libraries are found through /LIBPATH: and LIB, and /WHOLEARCHIVE: and
// /WHOLEARCHIVE extract their device code.
// RUN: rm -rf %t.libdir && mkdir %t.libdir
// RUN: llvm-ar rcs %t.libdir/foo.lib %t.win.obj
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --sycl-device-link --linker-path=link.exe /LIBPATH:%t.libdir \
// RUN:   /WHOLEARCHIVE:foo.lib -o dev.obj 2>&1 | FileCheck %s --check-prefix=WINLIB
// RUN: env LIB=%t.libdir clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc \
// RUN:   --dry-run --sycl-device-link --linker-path=link.exe /WHOLEARCHIVE \
// RUN:   foo.lib -o dev.obj 2>&1 | FileCheck %s --check-prefix=WINLIB
// WINLIB: --target=spirv64-unknown-unknown {{.*}}win-spirv64-unknown-unknown-generic{{.*}}.o --sycl-link{{$}}

// Missing inputs are reported: a /WHOLEARCHIVE: library, and a path that starts
// like a link.exe option.
// RUN: not clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --sycl-device-link --linker-path=link.exe /WHOLEARCHIVE:nope.lib \
// RUN:   -o dev.obj 2>&1 | FileCheck %s --check-prefix=MISSING -DFILE=nope.lib
// RUN: not clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --sycl-device-link --linker-path=link.exe /wholearchive-dir/missing.obj \
// RUN:   -o dev.obj 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISSING -DFILE=/wholearchive-dir/missing.obj

// In a host link without --sycl-device-link, the wrapped device object follows
// /OUT: in any case, and is added once even if the output is named twice.
// RUN: clang-linker-wrapper --host-triple=x86_64-pc-windows-msvc --dry-run \
// RUN:   --linker-path=link.exe %t.win.obj /OUT:app.exe -o app.exe 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WIN-LINK
// WIN-LINK: "link.exe" {{.*}}win.obj /OUT:app.exe {{[^ ]*}}app.exe.sycl.image.wrapper{{[^ ]*}}.o -o app.exe{{$}}
