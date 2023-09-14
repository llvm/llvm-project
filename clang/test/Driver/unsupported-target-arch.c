// Tests that clang does not crash with invalid architectures in target triples.
//
// RUN: not %clang --target=noarch-unknown-linux -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-LINUX %s
// CHECK-NOARCH-LINUX: error: unknown target triple 'noarch-unknown-linux'{{$}}
//
// RUN: not %clang --target=noarch-unknown-darwin -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-DARWIN %s
// CHECK-NOARCH-DARWIN: error: unknown target triple 'unknown-unknown-macosx{{.+}}'{{$}}
//
// RUN: not %clang --target=noarch-unknown-windows -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-WINDOWS %s
// CHECK-NOARCH-WINDOWS: error: unknown target triple 'noarch-unknown-windows-{{.+}}'{{$}}
//
// RUN: not %clang --target=noarch-unknown-freebsd -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-FREEBSD %s
// CHECK-NOARCH-FREEBSD: error: unknown target triple 'noarch-unknown-freebsd'{{$}}
//
// RUN: not %clang --target=noarch-unknown-netbsd -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-NETBSD %s
// CHECK-NOARCH-NETBSD: error: unknown target triple 'noarch-unknown-netbsd'{{$}}
//
// RUN: not %clang --target=noarch-unknown-nacl -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-NACL %s
// CHECK-NOARCH-NACL:  error: the target architecture 'noarch' is not supported by the target 'Native Client'

// RUN: not %clang --target=noarch-unknown-windows-gnu -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-MINGW %s
// CHECK-NOARCH-MINGW: error: unknown target triple 'noarch-unknown-windows-gnu'

// RUN: not %clang --target=noarch-unknown-windows-itanium -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-NOARCH-CROSSWINDOWS %s
// CHECK-NOARCH-CROSSWINDOWS: error: unknown target triple 'noarch-unknown-windows-itanium'

// RUN: not %clang --target=aarch64-none-eabi -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-AARCH64-INVALID-ENV %s
// CHECK-AARCH64-INVALID-ENV: warning: mismatch between architecture and environment in target triple 'aarch64-none-eabi'; did you mean 'aarch64-none-elf'? [-Winvalid-command-line-argument]{{$}}

// RUN: not %clang --target=aarch64_be-none-eabihf -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-AARCH64_BE-INVALID-ENV %s
// CHECK-AARCH64_BE-INVALID-ENV: warning: mismatch between architecture and environment in target triple 'aarch64_be-none-eabihf'; did you mean 'aarch64_be-none-elf'? [-Winvalid-command-line-argument]{{$}}

// RUN: not %clang --target=aarch64_32-none-eabi -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-AARCH64_32-INVALID-ENV %s
// CHECK-AARCH64_32-INVALID-ENV: warning: mismatch between architecture and environment in target triple 'aarch64_32-none-eabi'; did you mean 'aarch64_32-none-elf'? [-Winvalid-command-line-argument]{{$}}

// RUN: not %clang --target=arm-none-elf -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-ARM-INVALID-ENV %s
// CHECK-ARM-INVALID-ENV: warning: mismatch between architecture and environment in target triple 'arm-none-elf'; did you mean 'arm-none-eabi'? [-Winvalid-command-line-argument]{{$}}

// RUN: not %clang --target=armeb-none-elf -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-ARMEB-INVALID-ENV %s
// CHECK-ARMEB-INVALID-ENV: warning: mismatch between architecture and environment in target triple 'armeb-none-elf'; did you mean 'armeb-none-eabi'? [-Winvalid-command-line-argument]{{$}}

// RUN: not %clang --target=thumbv6m-none-elf -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-THUMB-INVALID-ENV %s
// CHECK-THUMB-INVALID-ENV: warning: mismatch between architecture and environment in target triple 'thumbv6m-none-elf'; did you mean 'thumbv6m-none-eabi'? [-Winvalid-command-line-argument]{{$}}

// RUN: not %clang --target=thumbeb-none-elf -o %t.o %s 2> %t.err
// RUN: FileCheck --input-file=%t.err --check-prefix=CHECK-THUMBEB-INVALID-ENV %s
// CHECK-THUMBEB-INVALID-ENV: warning: mismatch between architecture and environment in target triple 'thumbeb-none-elf'; did you mean 'thumbeb-none-eabi'? [-Winvalid-command-line-argument]{{$}}
