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
