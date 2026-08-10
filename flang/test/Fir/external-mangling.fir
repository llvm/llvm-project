// RUN: fir-opt --external-name-interop="append-underscore=true" %s | FileCheck %s --check-prefix=CHECK-UNDER
// RUN: fir-opt --external-name-interop="append-underscore=false" %s | FileCheck %s --check-prefix=CHECK-NOUNDER
// RUN: tco --external-name-interop="append-underscore=true" %s | FileCheck %s --check-prefix=CHECK-UNDER
// RUN: tco --external-name-interop="append-underscore=false" %s | FileCheck %s --check-prefix=CHECK-NOUNDER
// RUN: tco --external-name-interop="append-underscore=true" %s | tco --fir-to-llvm-ir | FileCheck %s --check-prefix=LLVMIR-UNDER
// RUN: tco --external-name-interop="append-underscore=false" %s | tco --fir-to-llvm-ir | FileCheck %s --check-prefix=LLVMIR-NOUNDER

func.func @_QPfoo() {
  %c0 = arith.constant 0 : index
  %0 = fir.address_of(@_QCa) : !fir.ref<!fir.array<4xi8>>
  %1 = fir.convert %0 : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<i32>
  %4 = fir.address_of(@_QC) : !fir.ref<!fir.array<4xi8>>
  %5 = fir.convert %4 : (!fir.ref<!fir.array<4xi8>>) -> !fir.ref<!fir.array<?xi8>>
  %6 = fir.coordinate_of %5, %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ref<f32>
  fir.call @_QPbar(%3) : (!fir.ref<i32>) -> ()
  fir.call @_QPbar2(%7) : (!fir.ref<f32>) -> ()
  return
}
fir.global common @_QCa(dense<0> : vector<4xi8>) : !fir.array<4xi8>
fir.global common @_QC(dense<0> : vector<4xi8>) : !fir.array<4xi8>
func.func private @_QPbar(!fir.ref<i32>)
func.func private @_QPbar2(!fir.ref<f32>)

// CHECK-UNDER: func @foo_
// CHECK-UNDER: %{{.*}} = fir.address_of(@a_) : !fir.ref<!fir.array<4xi8>>
// CHECK-UNDER: %{{.*}} = fir.address_of(@__BLNK__) : !fir.ref<!fir.array<4xi8>>
// CHECK-UNDER: fir.call @bar_
// CHECK-UNDER: fir.call @bar2_
// CHECK-UNDER: fir.global common @a_(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK-UNDER: fir.global common @__BLNK__(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK-UNDER: func private @bar_(!fir.ref<i32>)

// CHECK-NOUNDER: func @foo(
// CHECK-NOUNDER: %{{.*}} = fir.address_of(@a) : !fir.ref<!fir.array<4xi8>>
// CHECK-NOUNDER: %{{.*}} = fir.address_of(@__BLNK__) : !fir.ref<!fir.array<4xi8>>
// CHECK-NOUNDER: fir.call @bar(
// CHECK-NOUNDER: fir.call @bar2(
// CHECK-NOUNDER: fir.global common @a(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK-NOUNDER: fir.global common @__BLNK__(dense<0> : vector<4xi8>) : !fir.array<4xi8>
// CHECK-NOUNDER: func private @bar(!fir.ref<i32>)

// LLVMIR-UNDER: %{{.*}} = llvm.mlir.addressof @a_ : !llvm.ptr
// LLVMIR-UNDER: %{{.*}} = llvm.mlir.addressof @__BLNK__ : !llvm.ptr
// LLVMIR-UNDER: llvm.call @bar_(%{{.*}}) : (!llvm.ptr) -> ()
// LLVMIR-UNDER: llvm.call @bar2_(%{{.*}}) : (!llvm.ptr) -> ()

// LLVMIR-UNDER: llvm.mlir.global common @a_(dense<0> : vector<4xi8>) {{.*}} : !llvm.array<4 x i8>
// LLVMIR-UNDER: llvm.mlir.global common @__BLNK__(dense<0> : vector<4xi8>) {{.*}}  : !llvm.array<4 x i8>
// LLVMIR-UNDER: llvm.func @bar_(!llvm.ptr) attributes {fir.internal_name = "_QPbar", sym_visibility = "private"}
// LLVMIR-UNDER: llvm.func @bar2_(!llvm.ptr) attributes {fir.internal_name = "_QPbar2", sym_visibility = "private"}

// LLVMIR-NOUNDER: %{{.*}} = llvm.mlir.addressof @a : !llvm.ptr
// LLVMIR-NOUNDER: %{{.*}} = llvm.mlir.addressof @__BLNK__ : !llvm.ptr
// LLVMIR-NOUNDER: llvm.call @bar(%{{.*}}) : (!llvm.ptr) -> ()
// LLVMIR-NOUNDER: llvm.call @bar2(%{{.*}}) : (!llvm.ptr) -> ()

// LLVMIR-NOUNDER: llvm.mlir.global common @a(dense<0> : vector<4xi8>) {{.*}} : !llvm.array<4 x i8>
// LLVMIR-NOUNDER: llvm.mlir.global common @__BLNK__(dense<0> : vector<4xi8>) {{.*}}  : !llvm.array<4 x i8>
// LLVMIR-NOUNDER: llvm.func @bar(!llvm.ptr) attributes {fir.internal_name = "_QPbar", sym_visibility = "private"}
// LLVMIR-NOUNDER: llvm.func @bar2(!llvm.ptr) attributes {fir.internal_name = "_QPbar2", sym_visibility = "private"}

// ----- 

func.func @_QPcallee() {
  fir.call @_QPcallee() : () -> ()
  return
}

func.func @_QPcaller() {
  fir.call @_QPcallee() : () -> ()
  return 
}

// CHECK-UNDER: func @callee_
// CHECK-UNDER: fir.call @callee_
// CHECK-UNDER: func @caller_
// CHECK-UNDER: fir.call @callee_

// LLVMIR-UNDER: llvm.call @callee_() : () -> ()
// LLVMIR-UNDER: llvm.call @callee_() : () -> ()

// CHECK-NOUNDER: func @callee(
// CHECK-NOUNDER: fir.call @callee(
// CHECK-NOUNDER: func @caller(
// CHECK-NOUNDER: fir.call @callee(

// LLVMIR-NOUNDER: llvm.call @callee() : () -> ()
// LLVMIR-NOUNDER: llvm.call @callee() : () -> ()
