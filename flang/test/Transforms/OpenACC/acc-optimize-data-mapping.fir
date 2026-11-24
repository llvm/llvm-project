// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-optimize-data-mapping)" -split-input-file | FileCheck %s

// An array that is only read does not need to be copied back. A written one does.
func.func @test_array_read_and_write() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %one = arith.constant 1 : i32
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %ra = fir.alloca !fir.array<10xi32> {bindc_name = "readonly"}
  %readonly = fir.declare %ra(%shape) uniq_name("_QFEreadonly") : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  %wa = fir.alloca !fir.array<10xi32> {bindc_name = "written"}
  %written = fir.declare %wa(%shape) uniq_name("_QFEwritten") : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  %r = acc.copyin varPtr(%readonly : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("readonly") -> !fir.ref<!fir.array<10xi32>>
  %w = acc.copyin varPtr(%written : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("written") -> !fir.ref<!fir.array<10xi32>>
  acc.parallel dataOperands(%r, %w : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>) {
    %relem = fir.array_coor %r(%shape) %c1 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %val = fir.load %relem : !fir.ref<i32>
    %welem = fir.array_coor %w(%shape) %c1 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    fir.store %val to %welem : !fir.ref<i32>
    acc.yield
  }
  acc.copyout accPtr(%w : !fir.ref<!fir.array<10xi32>>) to varPtr(%written : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("written")
  acc.copyout accPtr(%r : !fir.ref<!fir.array<10xi32>>) to varPtr(%readonly : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("readonly")
  return
}

// The remaining copy is moved in front of the converted copyin in the data
// operands. The copy owns the mapping when the two name the same memory, so
// the copyout still copies it back. The operations themselves stay in place.
// CHECK-LABEL: func.func @test_array_read_and_write
// CHECK: %[[R:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>) implicit(true) name("readonly")
// CHECK: %[[W:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("written")
// CHECK: acc.parallel dataOperands(%[[W]], %[[R]] :
// CHECK: acc.copyout accPtr(%[[W]] : !fir.ref<!fir.array<10xi32>>) to varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("written")
// CHECK: acc.delete accPtr(%[[R]] : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copyin) implicit(true) name("readonly")

// -----

// Fortran does not allow an intent(in) dummy to alias an intent(out) dummy,
// so alias analysis treats x and y as unmodified and their implicit copies
// become copyins. A caller can still pass one actual for both, as in
// "call apply(x, y, x)". Moving the copies in front of the first converted
// copyin leaves result, which stays a copy, as the first data operand. That
// copy creates the mapping, so the
// result is still copied back. The operations themselves stay in place.
func.func @test_dummy_copyin_ordering(
    %arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "x"},
    %arg1: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "y"},
    %arg2: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "result"}) {
  %c1 = arith.constant 1 : index
  %scope = fir.dummy_scope : !fir.dscope
  %x = fir.declare %arg0 dummy_scope %scope uniq_name("_QFapplyEx") fortran_attrs<intent_in> : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> !fir.box<!fir.array<?xf64>>
  %y = fir.declare %arg1 dummy_scope %scope uniq_name("_QFapplyEy") fortran_attrs<intent_in> : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> !fir.box<!fir.array<?xf64>>
  %result = fir.declare %arg2 dummy_scope %scope uniq_name("_QFapplyEresult") fortran_attrs<intent_out> : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> !fir.box<!fir.array<?xf64>>
  %xc = acc.copyin var(%x : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("x") -> !fir.box<!fir.array<?xf64>>
  %yc = acc.copyin var(%y : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("y") -> !fir.box<!fir.array<?xf64>>
  %rc = acc.copyin var(%result : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("result") -> !fir.box<!fir.array<?xf64>>
  acc.parallel dataOperands(%xc, %yc, %rc : !fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>) {
    %xelem = fir.array_coor %xc %c1 : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
    %xval = fir.load %xelem : !fir.ref<f64>
    %yelem = fir.array_coor %yc %c1 : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
    %yval = fir.load %yelem : !fir.ref<f64>
    %sum = arith.addf %xval, %yval : f64
    %relem = fir.array_coor %rc %c1 : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
    fir.store %sum to %relem : !fir.ref<f64>
    acc.yield
  }
  acc.copyout accVar(%rc : !fir.box<!fir.array<?xf64>>) to var(%result : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("result")
  acc.copyout accVar(%yc : !fir.box<!fir.array<?xf64>>) to var(%y : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("y")
  acc.copyout accVar(%xc : !fir.box<!fir.array<?xf64>>) to var(%x : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("x")
  return
}

// CHECK-LABEL: func.func @test_dummy_copyin_ordering
// CHECK: %[[X:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xf64>>) implicit(true) name("x")
// CHECK: %[[Y:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xf64>>) implicit(true) name("y")
// CHECK: %[[RESULT:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("result")
// CHECK: acc.parallel dataOperands(%[[RESULT]], %[[X]], %[[Y]] :
// CHECK: acc.copyout accVar(%[[RESULT]] : !fir.box<!fir.array<?xf64>>) to var(%{{.*}} : !fir.box<!fir.array<?xf64>>) dataClause(acc_copy) implicit(true) name("result")
// CHECK: acc.delete accVar(%[[Y]] : !fir.box<!fir.array<?xf64>>) dataClause(acc_copyin) implicit(true) name("y")
// CHECK: acc.delete accVar(%[[X]] : !fir.box<!fir.array<?xf64>>) dataClause(acc_copyin) implicit(true) name("x")

// -----

// The data of an allocatable is written through its descriptor, which is
// itself only read. The copy back is kept for the written one.
func.func @test_allocatable_data_write() {
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f32
  %ra = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "readonly"}
  %readonly = fir.declare %ra uniq_name("_QFEreadonly") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %wa = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "written"}
  %written = fir.declare %wa uniq_name("_QFEwritten") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %r = acc.copyin varPtr(%readonly : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) dataClause(acc_copy) implicit(true) name("readonly") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  %w = acc.copyin varPtr(%written : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) dataClause(acc_copy) implicit(true) name("written") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  acc.parallel dataOperands(%r, %w : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %rbox = fir.load %r : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %raddr = fir.box_addr %rbox : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %relem = fir.coordinate_of %raddr, %c1 : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    %val = fir.load %relem : !fir.ref<f32>
    %wbox = fir.load %w : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %waddr = fir.box_addr %wbox : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
    %welem = fir.coordinate_of %waddr, %c1 : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    fir.store %val to %welem : !fir.ref<f32>
    acc.yield
  }
  acc.copyout accPtr(%w : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) to varPtr(%written : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) dataClause(acc_copy) implicit(true) name("written")
  acc.copyout accPtr(%r : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) to varPtr(%readonly : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) dataClause(acc_copy) implicit(true) name("readonly")
  return
}

// CHECK-LABEL: func.func @test_allocatable_data_write
// CHECK: %[[R:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) implicit(true) name("readonly")
// CHECK: %[[W:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) dataClause(acc_copy) implicit(true) name("written")
// CHECK-NOT: fir.box_addr
// CHECK: acc.parallel dataOperands(%[[W]], %[[R]] :
// CHECK: acc.copyout accPtr(%[[W]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) to varPtr(%{{.*}}) dataClause(acc_copy) implicit(true) name("written")
// CHECK: acc.delete accPtr(%[[R]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) dataClause(acc_copyin) implicit(true) name("readonly")

// -----

// A descriptor passed by value has its data written.
func.func @test_box_value_data_write(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.0 : f32
  %scope = fir.dummy_scope : !fir.dscope
  %x = fir.declare %arg0 dummy_scope %scope uniq_name("_QFEx") : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?xf32>>
  %c = acc.copyin var(%x : !fir.box<!fir.array<?xf32>>) dataClause(acc_copy) implicit(true) name("x") -> !fir.box<!fir.array<?xf32>>
  acc.parallel dataOperands(%c : !fir.box<!fir.array<?xf32>>) {
    %elem = fir.array_coor %c %c1 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
    fir.store %one to %elem : !fir.ref<f32>
    acc.yield
  }
  acc.copyout accVar(%c : !fir.box<!fir.array<?xf32>>) to var(%x : !fir.box<!fir.array<?xf32>>) dataClause(acc_copy) implicit(true) name("x")
  return
}

// CHECK-LABEL: func.func @test_box_value_data_write
// CHECK: %[[C:.*]] = acc.copyin var(%{{.*}} : !fir.box<!fir.array<?xf32>>) dataClause(acc_copy) implicit(true) name("x")
// CHECK: acc.copyout accVar(%[[C]] : !fir.box<!fir.array<?xf32>>) to var(%{{.*}} : !fir.box<!fir.array<?xf32>>) dataClause(acc_copy) implicit(true) name("x")

// -----

// A named constant cannot be modified, even by a procedure it is passed to.
func.func private @_QPfoo(!fir.ref<!fir.array<10xi32>>)
fir.global internal @_QFECparam constant : !fir.array<10xi32> {
  %0 = fir.zero_bits !fir.array<10xi32>
  fir.has_value %0 : !fir.array<10xi32>
}
func.func @test_named_constant() {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %addr = fir.address_of(@_QFECparam) : !fir.ref<!fir.array<10xi32>>
  %param = fir.declare %addr(%shape) uniq_name("_QFECparam") fortran_attrs<parameter> : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  %cparam = acc.copyin varPtr(%param : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("param") -> !fir.ref<!fir.array<10xi32>>
  acc.serial dataOperands(%cparam : !fir.ref<!fir.array<10xi32>>) {
    fir.call @_QPfoo(%cparam) : (!fir.ref<!fir.array<10xi32>>) -> ()
    acc.yield
  }
  acc.copyout accPtr(%cparam : !fir.ref<!fir.array<10xi32>>) to varPtr(%param : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("param")
  return
}

// CHECK-LABEL: func.func @test_named_constant
// CHECK: %[[PARAM:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>) implicit(true) name("param")
// CHECK: acc.delete accPtr(%[[PARAM]] : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copyin) implicit(true) name("param")

// -----

// An explicit copy is left unchanged. The async attribute of an implicit copy
// is kept.
func.func @test_explicit_and_async(%arg0: i32) {
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %ea = fir.alloca !fir.array<10xi32> {bindc_name = "explicit"}
  %explicit = fir.declare %ea(%shape) uniq_name("_QFEexplicit") : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  %aa = fir.alloca !fir.array<10xi32> {bindc_name = "async"}
  %async = fir.declare %aa(%shape) uniq_name("_QFEasync") : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
  %e = acc.copyin varPtr(%explicit : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) name("explicit") -> !fir.ref<!fir.array<10xi32>>
  %a = acc.copyin varPtr(%async : !fir.ref<!fir.array<10xi32>>) async(%arg0 : i32) dataClause(acc_copy) implicit(true) name("async") -> !fir.ref<!fir.array<10xi32>>
  acc.parallel dataOperands(%e, %a : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>) async(%arg0 : i32) {
    acc.yield
  }
  acc.copyout accPtr(%a : !fir.ref<!fir.array<10xi32>>) async(%arg0 : i32) to varPtr(%async : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) implicit(true) name("async")
  acc.copyout accPtr(%e : !fir.ref<!fir.array<10xi32>>) to varPtr(%explicit : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) name("explicit")
  return
}

// CHECK-LABEL: func.func @test_explicit_and_async
// CHECK: %[[E:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) name("explicit")
// CHECK: %[[A:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>) async(%{{.*}} : i32) implicit(true) name("async")
// CHECK: acc.delete accPtr(%[[A]] : !fir.ref<!fir.array<10xi32>>) async(%{{.*}} : i32) dataClause(acc_copyin) implicit(true) name("async")
// CHECK: acc.copyout accPtr(%[[E]] : !fir.ref<!fir.array<10xi32>>) to varPtr(%{{.*}} : !fir.ref<!fir.array<10xi32>>) dataClause(acc_copy) name("explicit")

// -----

// An implicit copy listed by acc.kernel_environment keeps its copyout.
func.func @test_kernel_environment() {
  %one = arith.constant 1 : i32
  %ra = fir.alloca i32 {bindc_name = "readonly"}
  %readonly = fir.declare %ra uniq_name("_QFEreadonly") : (!fir.ref<i32>) -> !fir.ref<i32>
  %wa = fir.alloca i32 {bindc_name = "written"}
  %written = fir.declare %wa uniq_name("_QFEwritten") : (!fir.ref<i32>) -> !fir.ref<i32>
  %r = acc.copyin varPtr(%readonly : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly") -> !fir.ref<i32>
  %w = acc.copyin varPtr(%written : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("written") -> !fir.ref<i32>
  acc.kernel_environment dataOperands(%r, %w : !fir.ref<i32>, !fir.ref<i32>) {
    acc.compute_region ins(%arg0 = %r, %arg1 = %w) : (!fir.ref<i32>, !fir.ref<i32>) {
      %unused = fir.load %arg0 : !fir.ref<i32>
      %unused2 = fir.load %arg1 : !fir.ref<i32>
      acc.yield
    } <{origin = "acc.parallel"}>
    acc.compute_region ins(%arg0 = %r, %arg1 = %w, %arg2 = %one) : (!fir.ref<i32>, !fir.ref<i32>, i32) {
      %unused = fir.load %arg0 : !fir.ref<i32>
      fir.store %arg2 to %arg1 : !fir.ref<i32>
      acc.yield
    } <{origin = "acc.parallel"}>
  }
  acc.copyout accPtr(%w : !fir.ref<i32>) to varPtr(%written : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("written")
  acc.copyout accPtr(%r : !fir.ref<i32>) to varPtr(%readonly : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly")
  return
}

// CHECK-LABEL: func.func @test_kernel_environment
// CHECK: %[[R:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly")
// CHECK: %[[W:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("written")
// CHECK: acc.copyout accPtr(%[[W]] : !fir.ref<i32>) to varPtr(%{{.*}} : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("written")
// CHECK: acc.copyout accPtr(%[[R]] : !fir.ref<i32>) to varPtr(%{{.*}} : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly")

// -----

// acc.data is not a region this pass handles, so a mapping it lists is copied
// back even when the compute constructs inside it only read the variable. A
// named constant does not need that and is still optimized.
fir.global internal @_QFECparam constant : i32 {
  %0 = arith.constant 0 : i32
  fir.has_value %0 : i32
}
func.func @test_multiple_owners() {
  %ra = fir.alloca i32 {bindc_name = "readonly"}
  %readonly = fir.declare %ra uniq_name("_QFEreadonly") : (!fir.ref<i32>) -> !fir.ref<i32>
  %addr = fir.address_of(@_QFECparam) : !fir.ref<i32>
  %param = fir.declare %addr uniq_name("_QFECparam") fortran_attrs<parameter> : (!fir.ref<i32>) -> !fir.ref<i32>
  %r = acc.copyin varPtr(%readonly : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly") -> !fir.ref<i32>
  %p = acc.copyin varPtr(%param : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("param") -> !fir.ref<i32>
  acc.data dataOperands(%r, %p : !fir.ref<i32>, !fir.ref<i32>) {
    acc.parallel dataOperands(%r, %p : !fir.ref<i32>, !fir.ref<i32>) {
      %unused = fir.load %r : !fir.ref<i32>
      %unused2 = fir.load %p : !fir.ref<i32>
      acc.yield
    }
    acc.parallel dataOperands(%r, %p : !fir.ref<i32>, !fir.ref<i32>) {
      %unused = fir.load %r : !fir.ref<i32>
      %unused2 = fir.load %p : !fir.ref<i32>
      acc.yield
    }
    acc.terminator
  }
  acc.copyout accPtr(%p : !fir.ref<i32>) to varPtr(%param : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("param")
  acc.copyout accPtr(%r : !fir.ref<i32>) to varPtr(%readonly : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly")
  return
}

// CHECK-LABEL: func.func @test_multiple_owners
// CHECK: %[[R:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly")
// CHECK: %[[P:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) implicit(true) name("param")
// CHECK: acc.delete accPtr(%[[P]] : !fir.ref<i32>) dataClause(acc_copyin) implicit(true) name("param")
// CHECK: acc.copyout accPtr(%[[R]] : !fir.ref<i32>) to varPtr(%{{.*}} : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("readonly")

// -----

// The copy of a is moved in front of the converted copyin of idx and the
// present of idx. The present still follows the copyin that creates its data.
func.func @test_copy_moved_before_present() {
  %ia = fir.alloca i32 {bindc_name = "idx"}
  %idx = fir.declare %ia uniq_name("_QFEidx") : (!fir.ref<i32>) -> !fir.ref<i32>
  %aa = fir.alloca i32 {bindc_name = "a"}
  %a = fir.declare %aa uniq_name("_QFEa") : (!fir.ref<i32>) -> !fir.ref<i32>
  %ic = acc.copyin varPtr(%idx : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("idx") -> !fir.ref<i32>
  %ip = acc.present varPtr(%idx : !fir.ref<i32>) implicit(true) name("idx") -> !fir.ref<i32>
  %ac = acc.copyin varPtr(%a : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("a") -> !fir.ref<i32>
  acc.parallel dataOperands(%ic, %ip, %ac : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
    %v = fir.load %ip : !fir.ref<i32>
    fir.store %v to %ac : !fir.ref<i32>
    acc.yield
  }
  acc.copyout accPtr(%ac : !fir.ref<i32>) to varPtr(%a : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("a")
  acc.copyout accPtr(%ic : !fir.ref<i32>) to varPtr(%idx : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("idx")
  return
}

// CHECK-LABEL: func.func @test_copy_moved_before_present
// CHECK: %[[IC:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) implicit(true) name("idx")
// CHECK: %[[IP:.*]] = acc.present varPtr(%{{.*}} : !fir.ref<i32>) implicit(true) name("idx")
// CHECK: %[[AC:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) dataClause(acc_copy) implicit(true) name("a")
// CHECK: acc.parallel dataOperands(%[[AC]], %[[IC]], %[[IP]] :
// CHECK: acc.delete accPtr(%[[IC]] : !fir.ref<i32>) dataClause(acc_copyin) implicit(true) name("idx")
