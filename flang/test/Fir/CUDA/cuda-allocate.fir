// RUN: fir-opt --cuf-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {

func.func @_QPsub1() {
  %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Ea"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %4:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %c1 = arith.constant 1 : index
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %9 = cuf.allocate %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  %10 = cuf.deallocate %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  cuf.free %4#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub1()
// CHECK: %[[DESC_RT_CALL:.*]] = fir.call @_FortranACUFAllocDesciptor(%{{.*}}, %{{.*}}, %{{.*}}) : (i64, !fir.ref<i8>, i32) -> !fir.ref<!fir.box<none>>
// CHECK: %[[DESC:.*]] = fir.convert %[[DESC_RT_CALL]] : (!fir.ref<!fir.box<none>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK: %[[DECL_DESC:.*]]:2 = hlfir.declare %[[DESC]] {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[DECL_DESC]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[DECL_DESC]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: %{{.*}} = fir.call @_FortranAAllocatableDeallocate(%[[BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
// CHECK: %[[BOX_NONE:.*]] = fir.convert %[[DECL_DESC]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK: fir.call @_FortranACUFFreeDesciptor(%[[BOX_NONE]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.ref<i8>, i32) -> none

// Check operations that should not be transformed yet.
func.func @_QPsub2() {
  %0 = cuf.alloc !fir.array<10xf32> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QMcuda_varFcuda_alloc_freeEa"} -> !fir.ref<!fir.array<10xf32>>
  cuf.free %0 : !fir.ref<!fir.array<10xf32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub2()
// CHECK: cuf.alloc !fir.array<10xf32>
// CHECK: cuf.free %{{.*}} : !fir.ref<!fir.array<10xf32>>

fir.global @_QMmod1Ea {data_attr = #cuf.cuda<device>} : !fir.box<!fir.heap<!fir.array<?xf32>>> {
    %0 = fir.zero_bits !fir.heap<!fir.array<?xf32>>
    %c0 = arith.constant 0 : index
    %1 = fir.shape %c0 : (index) -> !fir.shape<1>
    %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
    fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xf32>>>
}

func.func @_QPsub3() {
  %0 = fir.address_of(@_QMmod1Ea) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %1:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmod1Ea"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
  %2 = cuf.allocate %1#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  %3 = cuf.deallocate %1#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {data_attr = #cuf.cuda<device>} -> i32
  return
}

// CHECK-LABEL: func.func @_QPsub3()
// CHECK: cuf.allocate
// CHECK: cuf.deallocate

}


