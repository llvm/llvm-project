// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s
func.func @_QQmain() {
  %0 = fir.alloca !fir.box<!fir.array<10xi32>, volatile>
  %c1 = arith.constant 1 : index
  %c5_i32 = arith.constant 5 : i32
  %c10 = arith.constant 10 : index
  %1 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xi32>>
  %2 = fir.volatile_cast %1 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %3 = fir.address_of(@_QFEb) : !fir.ref<!fir.array<10xi32>>
  %4 = fir.volatile_cast %3 : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
  %5 = arith.subi %c10, %c1 : index
  %6 = arith.addi %5, %c1 : index
  %7 = arith.divsi %6, %c1 : index
  cf.br ^bb1(%c1, %7 : index, index)
^bb1(%8: index, %9: index):  // 2 preds: ^bb0, ^bb2
  %c0 = arith.constant 0 : index
  %10 = arith.cmpi sgt, %9, %c0 : index
  cf.cond_br %10, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %11 = fircg.ext_array_coor %4(%c10)<%8> : (!fir.ref<!fir.array<10xi32>, volatile>, index, index) -> !fir.ref<i32, volatile>
  fir.store %c5_i32 to %11 : !fir.ref<i32, volatile>
  %12 = arith.addi %8, %c1 overflow<nsw> : index
  %c1_0 = arith.constant 1 : index
  %13 = arith.subi %9, %c1_0 : index
  cf.br ^bb1(%12, %13 : index, index)
^bb3:  // pred: ^bb1
  %14 = fircg.ext_embox %4(%c10) : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.box<!fir.array<10xi32>, volatile>
  %15 = fircg.ext_embox %2(%c10) : (!fir.ref<!fir.array<10xi32>, volatile>, index) -> !fir.box<!fir.array<10xi32>, volatile>
  fir.store %15 to %0 : !fir.ref<!fir.box<!fir.array<10xi32>, volatile>>
  %16 = fir.address_of(@_QQclX8b2af12247fe2a74a66d92bb35ca5038) : !fir.ref<!fir.char<1,13>>
  %c3_i32 = arith.constant 3 : i32
  %17 = fir.convert %0 : (!fir.ref<!fir.box<!fir.array<10xi32>, volatile>>) -> !fir.ref<!fir.box<none>>
  %18 = fir.volatile_cast %14 : (!fir.box<!fir.array<10xi32>, volatile>) -> !fir.box<!fir.array<10xi32>>
  %19 = fir.convert %18 : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %20 = fir.convert %16 : (!fir.ref<!fir.char<1,13>>) -> !fir.ref<i8>
  fir.call @_FortranAAssign(%17, %19, %20, %c3_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
  return
}
fir.global internal @_QFEa : !fir.array<10xi32> {
  %0 = fir.zero_bits !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}
fir.global internal @_QFEb : !fir.array<10xi32> {
  %0 = fir.zero_bits !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}
func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime}
fir.global linkonce @_QQclX8b2af12247fe2a74a66d92bb35ca5038 constant : !fir.char<1,13> {
  %0 = fir.string_lit "filename.f90\00"(13) : !fir.char<1,13>
  fir.has_value %0 : !fir.char<1,13>
}
// CHECK-LABEL:   llvm.func @_QQmain() {
// CHECK:           llvm.store volatile %{{.+}}, %{{.+}} : i32, !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%{{.+}}, %{{.+}}, %{{.+}}) <{isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
