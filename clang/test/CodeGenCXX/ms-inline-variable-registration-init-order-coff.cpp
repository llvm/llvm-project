// REQUIRES: x86-registered-target
// XFAIL: *
// RUN: split-file %S/ms-inline-variable-registration-init-order.cpp %t
// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++20 -emit-obj -o %t/mongod.obj %t/mongod.cpp
// RUN: %clang_cc1 -triple x86_64-windows-msvc -std=c++20 -emit-obj -o %t/mongos.obj %t/mongos.cpp
// RUN: llvm-objdump -r %t/mongod.obj %t/mongos.obj | FileCheck %s --check-prefixes=MONGOD,MONGOS

// The namespace-scope MongoDB reproducer has the correct llvm.global_ctors
// order, but still fails after fixing inline static data member initialization.
// WinCOFFObjectWriter::assignSectionNumbers moves associative COMDAT sections
// after ordinary sections, reversing the two .CRT$XCU entries. lld orders CRT
// contributions from the same object by section number, so registration runs
// before options have been initialized. Keep this expected failure separate
// from the passing IR test until object emission preserves the required order.
// Inspect the actual object, since assembly and IR retain the source order.

// MONGOD-NOT: _GLOBAL__sub_I_mongod.cpp
// MONGOD: RELOCATION RECORDS FOR [.CRT$XCU]:
// MONGOD-NEXT: OFFSET {{ *}}TYPE {{ *}}VALUE
// MONGOD-NEXT: {{[0-9a-f]+}} IMAGE_REL_AMD64_ADDR64 {{ *}}??__Eoptions@@YAXXZ
// MONGOD: RELOCATION RECORDS FOR [.CRT$XCU]:
// MONGOD-NEXT: OFFSET {{ *}}TYPE {{ *}}VALUE
// MONGOD-NEXT: {{[0-9a-f]+}} IMAGE_REL_AMD64_ADDR64 {{ *}}_GLOBAL__sub_I_mongod.cpp

// MONGOS-NOT: _GLOBAL__sub_I_mongos.cpp
// MONGOS: RELOCATION RECORDS FOR [.CRT$XCU]:
// MONGOS-NEXT: OFFSET {{ *}}TYPE {{ *}}VALUE
// MONGOS-NEXT: {{[0-9a-f]+}} IMAGE_REL_AMD64_ADDR64 {{ *}}??__Eoptions@@YAXXZ
// MONGOS: RELOCATION RECORDS FOR [.CRT$XCU]:
// MONGOS-NEXT: OFFSET {{ *}}TYPE {{ *}}VALUE
// MONGOS-NEXT: {{[0-9a-f]+}} IMAGE_REL_AMD64_ADDR64 {{ *}}_GLOBAL__sub_I_mongos.cpp
