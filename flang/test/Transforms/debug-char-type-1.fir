// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  fir.global @_QMhelperEstr1 : !fir.char<1,40> {
    %0 = fir.zero_bits !fir.char<1,40>
    fir.has_value %0 : !fir.char<1,40>
  } loc(#loc1)
  fir.global @_QMhelperEstr2 : !fir.char<4,20> {
    %0 = fir.zero_bits !fir.char<4,20>
    fir.has_value %0 : !fir.char<4,20>
  } loc(#loc1)
  fir.global @_QMhelperEstr3 : !fir.box<!fir.heap<!fir.char<1,?>>> {
    %c0 = arith.constant 0 : index
    %0 = fir.zero_bits !fir.heap<!fir.char<1,?>>
    %1 = fir.embox %0 typeparams %c0 : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
    fir.has_value %1 : !fir.box<!fir.heap<!fir.char<1,?>>>
  } loc(#loc1)
}
#loc1 = loc("string.f90":1:1)

// CHECK-DAG: #[[TY1:.*]] = #llvm.di_string_type<tag = DW_TAG_string_type, name = "", sizeInBits = 320, encoding = DW_ATE_ASCII>
// CHECK-DAG: #llvm.di_global_variable<{{.*}}name = "str1"{{.*}}type = #[[TY1]]{{.*}}>
// CHECK-DAG: #[[TY2:.*]] = #llvm.di_string_type<tag = DW_TAG_string_type, name = "", sizeInBits = 640, encoding = DW_ATE_UCS>
// CHECK-DAG: #llvm.di_global_variable<{{.*}}name = "str2"{{.*}}type = #[[TY2]]{{.*}}>
// CHECK-DAG: #[[TY3:.*]] = #llvm.di_string_type<tag = DW_TAG_string_type{{.*}}stringLengthExp = <[DW_OP_push_object_address, DW_OP_plus_uconst(8)]>, stringLocationExp = <[DW_OP_push_object_address, DW_OP_deref]>, encoding = DW_ATE_ASCII>
// CHECK-DAG: #llvm.di_global_variable<{{.*}}name = "str3"{{.*}}type = #[[TY3]]{{.*}}>
