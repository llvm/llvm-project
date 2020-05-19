// RUN: %clang --target=x86_64-pc-linux -fcf-protection  -include cet.h -c %s -o - | llvm-readelf -n | FileCheck %s
// RUN: %clang --target=x86_64-pc-linux -include cet.h -c %s -o - | llvm-readelf -S | FileCheck %s --check-prefixes=NOCET
// RUN: %clang --target=x86_64-pc-linux -include cet.h -S %s -o - | FileCheck %s --check-prefixes=NOENDBR
// RUN: %clang --target=x86_64-pc-linux -fcf-protection  -include cet.h -S %s -o - | FileCheck %s --check-prefixes=ENDBR64

// RUN: %clang --target=i386-pc-linux -fcf-protection  -include cet.h -c %s -o - | llvm-readelf -n | FileCheck %s
// RUN: %clang --target=i386-pc-linux -include cet.h -c %s -o - | llvm-readelf -S | FileCheck %s --check-prefixes=NOCET
// RUN: %clang --target=i386-pc-linux -include cet.h -S %s -o - | FileCheck %s --check-prefixes=NOENDBR
// RUN: %clang --target=i386-pc-linux -fcf-protection  -include cet.h -S %s -o - | FileCheck %s --check-prefixes=ENDBR32

// CHECK: IBT, SHSTK

// NOCET:     Section Headers
// NOCET-NOT: .note.gnu.property

// NOENDBR:   foo
// NOENDBR-NOT: endbr

// ENDBR64: endbr64
// ENDBR32: endbr32
        .text
        .globl  foo
        .type   foo, @function
foo:
        _CET_ENDBR
        ret
