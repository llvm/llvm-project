// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-pch -fmodule-format=obj %S/pchpch1.h -o - | llvm-readelf --sections - | FileCheck %s

// Ensure the serialized AST is emitted via llvm.raw.sections metadata into
// a __clangast section with 8-byte alignment.

// CHECK: __clangast        PROGBITS  {{[0-9a-f]+}} {{[0-9a-f]+}} {{[0-9a-f]+}} 00   A  0   0  8
