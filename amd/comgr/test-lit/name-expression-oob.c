// COM: Regression test for ROCM-26343. A name-expression code object whose
// COM: .rela.dyn r_addend points outside .rodata must be rejected with an error
// COM: rather than triggering an out-of-bounds read in
// COM: amd_comgr_populate_name_expression_map. A successful run (no crash, exit
// COM: 0) that prints "RESULT: ERROR" confirms the offset is bounds-checked.

// RUN: %yaml2obj %S/name-expression-oob.yaml -o %t.o
// RUN: name-expression-map %t.o | %FileCheck %s

// CHECK: RESULT: ERROR
