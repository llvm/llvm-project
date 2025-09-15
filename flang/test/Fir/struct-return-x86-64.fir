// Test X86-64 ABI rewrite of struct returned by value (BIND(C), VALUE derived types).
// RUN: fir-opt --target-rewrite %s | FileCheck %s

!fits_in_reg = !fir.type<t1{i:f32,j:i32,k:f32}>
!too_big = !fir.type<t2{i:!fir.array<5xf32>}>

module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {

  func.func private @test_inreg() -> !fits_in_reg
  func.func @test_call_inreg(%arg0: !fir.ref<!fits_in_reg>) {
    %0 = fir.call @test_inreg() : () -> !fits_in_reg
    fir.store %0 to %arg0 : !fir.ref<!fits_in_reg>
    return
  }
  func.func @test_addr_of_inreg() -> (() -> ()) {
    %0 = fir.address_of(@test_inreg) : () -> !fits_in_reg
    %1 = fir.convert %0 : (() -> !fits_in_reg) -> (() -> ())
    return %1 : () -> ()
  }
  func.func @test_addr_of_inreg_2() -> (() -> !fits_in_reg) {
    %0 = fir.address_of(@test_inreg) : () -> !fits_in_reg
    return %0 : () -> !fits_in_reg
  }
  func.func @test_dispatch_inreg(%arg0: !fir.ref<!fits_in_reg>, %arg1: !fir.class<!fir.type<somet>>) {
    %0 = fir.dispatch "bar"(%arg1 : !fir.class<!fir.type<somet>>) (%arg1 : !fir.class<!fir.type<somet>>) -> !fits_in_reg {pass_arg_pos = 0 : i32}
    fir.store %0 to %arg0 : !fir.ref<!fits_in_reg>
    return
  }

  func.func private @test_sret() -> !too_big
  func.func @test_call_sret(%arg0: !fir.ref<!too_big>) {
    %0 = fir.call @test_sret() : () -> !too_big
    fir.store %0 to %arg0 : !fir.ref<!too_big>
    return
  }
  func.func @test_addr_of_sret() -> (() -> ()) {
    %0 = fir.address_of(@test_sret) : () -> !too_big
    %1 = fir.convert %0 : (() -> !too_big) -> (() -> ())
    return %1 : () -> ()
  }
  func.func @test_dispatch_sret(%arg0: !fir.ref<!too_big>, %arg1: !fir.class<!fir.type<somet>>) {
    %0 = fir.dispatch "bar"(%arg1 : !fir.class<!fir.type<somet>>) (%arg1 : !fir.class<!fir.type<somet>>) -> !too_big {pass_arg_pos = 0 : i32}
    fir.store %0 to %arg0 : !fir.ref<!too_big>
    return
  }
  func.func private @test_fp_80() -> !fir.type<t3{i:f80}>
  func.func private @test_complex_80() -> !fir.type<t4{i:complex<f80>}>
  func.func private @test_two_fp_80() -> !fir.type<t5{i:f80,j:f80}>
  func.func private @test_fp128() -> !fir.type<t6{i:f128}>
}

// CHECK-LABEL:   func.func private @test_inreg() -> tuple<i64, f32>

// CHECK-LABEL:   func.func @test_call_inreg(
// CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>) {
// CHECK:           %[[VAL_1:.*]] = fir.call @test_inreg() : () -> tuple<i64, f32>
// CHECK:           %[[VAL_2:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = fir.alloca tuple<i64, f32>
// CHECK:           fir.store %[[VAL_1]] to %[[VAL_3]] : !fir.ref<tuple<i64, f32>>
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<tuple<i64, f32>>) -> !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
// CHECK:           llvm.intr.stackrestore %[[VAL_2]] : !llvm.ptr
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_0]] : !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_addr_of_inreg() -> (() -> ()) {
// CHECK:           %[[VAL_0:.*]] = fir.address_of(@test_inreg) : () -> tuple<i64, f32>
// CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (() -> tuple<i64, f32>) -> (() -> !fir.type<t1{i:f32,j:i32,k:f32}>)
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (() -> !fir.type<t1{i:f32,j:i32,k:f32}>) -> (() -> ())
// CHECK:           return %[[VAL_2]] : () -> ()
// CHECK:         }

// CHECK-LABEL:   func.func @test_addr_of_inreg_2() -> (() -> !fir.type<t1{i:f32,j:i32,k:f32}>) {
// CHECK:           %[[VAL_0:.*]] = fir.address_of(@test_inreg) : () -> tuple<i64, f32>
// CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (() -> tuple<i64, f32>) -> (() -> !fir.type<t1{i:f32,j:i32,k:f32}>)
// CHECK:           return %[[VAL_1]] : () -> !fir.type<t1{i:f32,j:i32,k:f32}>
// CHECK:         }

// CHECK-LABEL:   func.func @test_dispatch_inreg(
// CHECK-SAME:                                   %[[VAL_0:.*]]: !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: !fir.class<!fir.type<somet>>) {
// CHECK:           %[[VAL_2:.*]] = fir.dispatch "bar"(%[[VAL_1]] : !fir.class<!fir.type<somet>>) (%[[VAL_1]] : !fir.class<!fir.type<somet>>) -> tuple<i64, f32> {pass_arg_pos = 0 : i32}
// CHECK:           %[[VAL_3:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = fir.alloca tuple<i64, f32>
// CHECK:           fir.store %[[VAL_2]] to %[[VAL_4]] : !fir.ref<tuple<i64, f32>>
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<tuple<i64, f32>>) -> !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
// CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
// CHECK:           llvm.intr.stackrestore %[[VAL_3]] : !llvm.ptr
// CHECK:           fir.store %[[VAL_6]] to %[[VAL_0]] : !fir.ref<!fir.type<t1{i:f32,j:i32,k:f32}>>
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @test_sret(!fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t2{i:!fir.array<5xf32>}>})

// CHECK-LABEL:   func.func @test_call_sret(
// CHECK-SAME:                              %[[VAL_0:.*]]: !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>) {
// CHECK:           %[[VAL_1:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.type<t2{i:!fir.array<5xf32>}>
// CHECK:           fir.call @test_sret(%[[VAL_2]]) : (!fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>> {llvm.align = 8 : i32, llvm.sret = !fir.type<t2{i:!fir.array<5xf32>}>}) -> ()
// CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>) -> !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>
// CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>
// CHECK:           llvm.intr.stackrestore %[[VAL_1]] : !llvm.ptr
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @test_addr_of_sret() -> (() -> ()) {
// CHECK:           %[[VAL_0:.*]] = fir.address_of(@test_sret) : (!fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>) -> ()
// CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : ((!fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>) -> ()) -> (() -> !fir.type<t2{i:!fir.array<5xf32>}>)
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (() -> !fir.type<t2{i:!fir.array<5xf32>}>) -> (() -> ())
// CHECK:           return %[[VAL_2]] : () -> ()
// CHECK:         }

// CHECK-LABEL:   func.func @test_dispatch_sret(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.class<!fir.type<somet>>) {
// CHECK:           %[[VAL_2:.*]] = llvm.intr.stacksave : !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.type<t2{i:!fir.array<5xf32>}>
// CHECK:           fir.dispatch "bar"(%[[VAL_1]] : !fir.class<!fir.type<somet>>) (%[[VAL_3]], %[[VAL_1]] : !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>, !fir.class<!fir.type<somet>>) {pass_arg_pos = 1 : i32}
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>) -> !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>
// CHECK:           llvm.intr.stackrestore %[[VAL_2]] : !llvm.ptr
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_0]] : !fir.ref<!fir.type<t2{i:!fir.array<5xf32>}>>
// CHECK:           return
// CHECK:         }


// CHECK: func.func private @test_fp_80() -> f80
// CHECK: func.func private @test_complex_80(!fir.ref<!fir.type<t4{i:complex<f80>}>> {llvm.align = 16 : i32, llvm.sret = !fir.type<t4{i:complex<f80>}>})
// CHECK: func.func private @test_two_fp_80(!fir.ref<!fir.type<t5{i:f80,j:f80}>> {llvm.align = 16 : i32, llvm.sret = !fir.type<t5{i:f80,j:f80}>})
// CHECK: func.func private @test_fp128() -> f128
