// RUN: fir-opt --external-name-interop %s | FileCheck %s
// RUN: tco --external-name-interop %s | FileCheck %s
// RUN: tco --external-name-interop %s | tco --fir-to-llvm-ir | FileCheck %s --check-prefix=LLVMIR

func.func @_QPfoo() {  
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QBa) : !fir.ref<!fir.array<4xi8>>
  %1 = fir.convert %0 : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<i32>
  %4 = fir.address_of(@_QB) : !fir.ref<!fir.array<4xi8>>
  %5 = fir.convert %4 : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %6 = fir.coordinate_of %5, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  fir.call @_QPbar(%3) : (!fir.ref<i32>) -> ()
  fir.call @_QPbar2(%7) : (!fir.ref<f32>) -> ()
  return
}
fir.global common @_QBa(dense<0> : vector<4xi8>) : !fir.array<4xi8>
fir.global common @_QB(dense<0> : vector<4xi8>) : !fir.array<4xi8>
func.func private @_QPbar(!fir.ref<i32>)
func.func private @_QPbar2(!fir.ref<f32>)

// CHECK: func @foo_
// CHECK: %{{.*}} = fir.address_of(@a_) : !fir.ref<!fir.array<4xi8>>
// CHECK: %{{.*}} = fir.address_of(@__BLNK__) : !fir.ref<!fir.array<4xi8>>
// CHECK: fir.call @bar_
// CHECK: fir.call @bar2_
// CHECK: fir.global common @a_(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK: fir.global common @__BLNK__(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK: func private @bar_(!fir.ref<i32>)

// LLVMIR: %{{.*}} = llvm.mlir.addressof @a_ : !llvm.ptr<array<4 x i8>>
// LLVMIR: %{{.*}} = llvm.mlir.addressof @__BLNK__ : !llvm.ptr<array<4 x i8>>
// LLVMIR: llvm.call @bar_(%{{.*}}) : (!llvm.ptr<i32>) -> ()
// LLVMIR: llvm.call @bar2_(%{{.*}}) : (!llvm.ptr<f32>) -> ()

// LLVMIR: llvm.mlir.global common @a_(dense<0> : vector<4xi8>) : !llvm.array<4 x i8>
// LLVMIR: llvm.mlir.global common @__BLNK__(dense<0> : vector<4xi8>) : !llvm.array<4 x i8>
// LLVMIR: llvm.func @bar_(!llvm.ptr<i32>) attributes {sym_visibility = "private"}
// LLVMIR: llvm.func @bar2_(!llvm.ptr<f32>) attributes {sym_visibility = "private"}
