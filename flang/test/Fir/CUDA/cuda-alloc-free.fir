// RUN: fir-opt --cuf-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {

func.func @_QPsub1() {
  %0 = cuf.alloc i32 {bindc_name = "idev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Eidev"} -> !fir.ref<i32>
  %1:2 = hlfir.declare %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Eidev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  cuf.free %1#1 : !fir.ref<i32> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub1()
// CHECK: %[[BYTES:.*]] = fir.convert %c4{{.*}} : (index) -> i64
// CHECK: %[[ALLOC:.*]] = fir.call @_FortranACUFMemAlloc(%[[BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) {cuf.data_attr = #cuf.cuda<device>} : (i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: %[[CONV:.*]] = fir.convert %3 : (!fir.llvm_ptr<i8>) -> !fir.ref<i32>
// CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[CONV]] {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub1Eidev"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[DEVPTR:.*]] = fir.convert %[[DECL]]#1 : (!fir.ref<i32>) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFMemFree(%[[DEVPTR]], %c0{{.*}}, %{{.*}}, %{{.*}}) : (!fir.llvm_ptr<i8>, i32, !fir.ref<i8>, i32) -> ()

func.func @_QPsub2() {
  %0 = cuf.alloc !fir.array<10xf32> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QMcuda_varFcuda_alloc_freeEa"} -> !fir.ref<!fir.array<10xf32>>
  cuf.free %0 : !fir.ref<!fir.array<10xf32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub2()
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c4{{.*}} : index
// CHECK: %[[CONV_BYTES:.*]] = fir.convert %[[BYTES]] : (index) -> i64 
// CHECK: %{{.*}} = fir.call @_FortranACUFMemAlloc(%[[CONV_BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) {cuf.data_attr = #cuf.cuda<device>} : (i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFMemFree

func.func @_QPsub3(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}, %arg1: !fir.ref<i32> {fir.bindc_name = "m"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFsub3En"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFsub3Em"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> i64
  %5 = fir.convert %4 : (i64) -> index
  %c0 = arith.constant 0 : index
  %6 = arith.cmpi sgt, %5, %c0 : index
  %7 = arith.select %6, %5, %c0 : index
  %8 = fir.load %2#0 : !fir.ref<i32>
  %9 = fir.convert %8 : (i32) -> i64
  %10 = fir.convert %9 : (i64) -> index
  %c0_0 = arith.constant 0 : index
  %11 = arith.cmpi sgt, %10, %c0_0 : index
  %12 = arith.select %11, %10, %c0_0 : index
  %13 = cuf.alloc !fir.array<?x?xi32>, %7, %12 : index, index {bindc_name = "idev", data_attr = #cuf.cuda<device>, uniq_name = "_QFsub3Eidev"} -> !fir.ref<!fir.array<?x?xi32>>
  %14 = fir.shape %7, %12 : (index, index) -> !fir.shape<2>
  %15:2 = hlfir.declare %13(%14) {data_attr = #cuf.cuda<device>, uniq_name = "_QFsub3Eidev"} : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>)
  cuf.free %15#1 : !fir.ref<!fir.array<?x?xi32>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPsub3
// CHECK: %[[N:.*]] = arith.select 
// CHECK: %[[M:.*]] = arith.select
// CHECK: %[[NBELEM:.*]] = arith.muli %[[N]], %[[M]] : index
// CHECK: %[[BYTES:.*]] = arith.muli %[[NBELEM]], %c4{{.*}} : index
// CHECK: %[[CONV_BYTES:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: %{{.*}} = fir.call @_FortranACUFMemAlloc(%[[CONV_BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) {cuf.data_attr = #cuf.cuda<device>} : (i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranACUFMemFree

func.func @_QPtest_type() {
  %0 = cuf.alloc !fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFtest_typeEa"} -> !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>
  %1 = fir.declare %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFtest_typeEa"} : (!fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>) -> !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>>
  cuf.free %1 : !fir.ref<!fir.type<_QMbarTcmplx{id:i32,c:complex<f32>}>> {data_attr = #cuf.cuda<device>}
  return
}

// CHECK-LABEL: func.func @_QPtest_type()
// CHECK: %[[BYTES:.*]] = arith.constant 12 : index
// CHECK: %[[CONV_BYTES:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFMemAlloc(%[[CONV_BYTES]], %c0{{.*}}, %{{.*}}, %{{.*}}) {cuf.data_attr = #cuf.cuda<device>} : (i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>

gpu.module @cuda_device_mod {
  gpu.func @_QMalloc() kernel {
    %0 = cuf.alloc !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QMallocEa"} -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    gpu.return 
  }
}

// CHECK-LABEL: gpu.func @_QMalloc() kernel
// CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", cuf.data_attr = #cuf.cuda<device>, uniq_name = "_QMallocEa"}

func.func @_QQalloc_char() attributes {fir.bindc_name = "alloc_char"} {
  %c1 = arith.constant 1 : index
  %0 = cuf.alloc !fir.array<10x!fir.char<1>>(%c1 : index) {bindc_name = "a", data_attr = #cuf.cuda<device>, uniq_name = "_QFEa"} -> !fir.ref<!fir.array<10x!fir.char<1>>>
  return
}

// CHECK-LABEL: func.func @_QQalloc_char()
// CHECK: %[[BYTES:.*]] = arith.muli %c10{{.*}}, %c1{{.*}} : index
// CHECK: %[[BYTES_CONV:.*]] = fir.convert %[[BYTES]] : (index) -> i64
// CHECK: fir.call @_FortranACUFMemAlloc(%[[BYTES_CONV]], %c0{{.*}}, %{{.*}}, %{{.*}}) {cuf.data_attr = #cuf.cuda<device>} : (i64, i32, !fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>

} // end module
