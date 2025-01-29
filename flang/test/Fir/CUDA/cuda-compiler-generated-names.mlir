// RUN: fir-opt --split-input-file --compiler-generated-names --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu type-descriptors-renamed-for-assembly=true" %s | FileCheck %s

module @mod1 attributes {gpu.container} {
  gpu.module @gpu1 {
    fir.global linkonce @_QMtest_dinitE.dt.tseq constant : i8

    func.func @embox1(%arg0: !fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) {
      %0 = fir.embox %arg0() : (!fir.ref<!fir.type<_QMtest_dinitTtseq{i:i32}>>) -> !fir.box<!fir.type<_QMtest_dinitTtseq{i:i32}>>
      return
    }

    fir.global @_QQdefault.nonTbpDefinedIoTable constant : tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1> {
      %true = arith.constant true
      %c0_i64 = arith.constant 0 : i64
      %0 = fir.undefined tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
      %1 = fir.insert_value %0, %c0_i64, [0 : index] : (tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>, i64) -> tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
      %2 = fir.zero_bits !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>
      %3 = fir.insert_value %1, %2, [1 : index] : (tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>) -> tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
      %4 = fir.insert_value %3, %true, [2 : index] : (tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>, i1) -> tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
      fir.has_value %4 : tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>
    }

    func.func @special() {
      %0 = fir.address_of(@_QQdefault.nonTbpDefinedIoTable) : !fir.ref<tuple<i64, !fir.ref<!fir.array<0xtuple<!fir.ref<none>, !fir.ref<none>, i32, i1>>>, i1>>
      return
    }
  }
}

// CHECK-LABEL: gpu.module @gpu1
// CHECK: llvm.mlir.global linkonce constant @_QMtest_dinitEXdtXtseq
// CHECK: llvm.mlir.addressof @_QMtest_dinitEXdtXtseq : !llvm.ptr

// CHECK: llvm.mlir.global external constant @_QQdefaultXnonTbpDefinedIoTable()
// CHECK: llvm.mlir.addressof @_QQdefaultXnonTbpDefinedIoTable
