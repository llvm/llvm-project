// Tests that LLVM conversion for declare mapper types is applied to
// `omp.declare_mapper` ops and correctly converts the declare mapper
// type when it's a box.

// RUN: fir-opt --convert-hlfir-to-fir --cg-rewrite --fir-to-llvm-ir %s -o - | \
// RUN: FileCheck %s

module {
  omp.declare_mapper @_QQFdeclare_mapper_1my_type_omp_default_mapper : !fir.box<!fir.heap<!fir.array<?xi32>>> {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
    %0 = fir.box_offset %arg0 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
    %1 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%0 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
    %2 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(always, to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "var%values(1:var%num_vals)"}
    omp.declare_mapper.info map_entries(%2, %1 :!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
  }
}

// CHECK:  omp.declare_mapper @_QQFdeclare_mapper_1my_type_omp_default_mapper : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {
// CHECK:  ^bb0(%arg0: !llvm.ptr):
