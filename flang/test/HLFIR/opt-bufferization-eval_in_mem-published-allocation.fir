// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// Test that the result of a call is evaluated in place into an allocatable
// whose storage is reallocated before the assignment only when the call
// cannot access the allocatable. The new storage is stored into the
// descriptor of the allocatable, so it is the data of the allocatable.
//
// module m
//   integer, allocatable :: ga(:)
// contains
//   function indep(n) result(r)
//     integer, intent(in) :: n
//     integer :: r(n), i
//     do i = 1, n
//       r(i) = 10 * i
//     end do
//   end function
//   function reads_arg(x) result(r)
//     integer, intent(in) :: x(:)
//     integer :: r(size(x) + 1), i
//     r(1) = 0
//     do i = 1, size(x)
//       r(i + 1) = r(i) + x(i)
//     end do
//   end function
//   function reads_module(n) result(r)
//     integer, intent(in) :: n
//     integer :: r(n), i
//     do i = 1, n
//       r(i) = ga(min(i, size(ga))) + i
//     end do
//   end function
//   subroutine positive(r)
//     integer, intent(out) :: r
//     integer, allocatable :: a(:)
//     a = indep(3)
//     r = sum(a)
//   end subroutine
//   subroutine via_argument(r)
//     integer, intent(out) :: r
//     integer, allocatable :: a(:)
//     a = [1, 2, 3]
//     a = reads_arg(a)
//     r = sum(a * [1, 2, 3, 4])
//   end subroutine
//   subroutine via_module(r)
//     integer, intent(out) :: r
//     allocate(ga(2))
//     ga = [5, 6]
//     ga = reads_module(4)
//     r = sum(ga * [1, 2, 3, 4])
//     deallocate(ga)
//   end subroutine
//   subroutine via_pointer(r)
//     integer, intent(out) :: r
//     integer, allocatable, target :: a(:)
//     integer, pointer :: p(:)
//     a = [1, 2, 3]
//     p => a
//     a = reads_arg(p)
//     r = sum(a * [1, 2, 3, 4])
//   end subroutine
//   subroutine via_host(r)
//     integer, intent(out) :: r
//     integer, allocatable :: a(:)
//     a = [1, 2, 3]
//     call inner()
//     r = sum(a * [1, 2, 3, 4])
//   contains
//     subroutine inner()
//       a = reads_arg(a)
//     end subroutine
//   end subroutine
//   recursive subroutine save_recursive(r)
//     integer, intent(out) :: r
//     integer, allocatable, save :: a(:)
//     a = indep(3)
//     r = sum(a)
//   end subroutine
//   non_recursive subroutine save_non_recursive(r)
//     integer, intent(out) :: r
//     integer, allocatable, save :: a(:)
//     a = indep(3)
//     r = sum(a)
//   end subroutine
// end module
//
// The code after each assignment is not included.

fir.global @_QMmEga : !fir.box<!fir.heap<!fir.array<?xi32>>> {
  %0 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}
fir.global internal @_QMmFsave_non_recursiveEa : !fir.box<!fir.heap<!fir.array<?xi32>>> {
  %0 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}
fir.global internal @_QMmFsave_recursiveEa : !fir.box<!fir.heap<!fir.array<?xi32>>> {
  %0 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c0 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.has_value %2 : !fir.box<!fir.heap<!fir.array<?xi32>>>
}
fir.global internal @_QQro.2xi4.2(dense<[5, 6]> : tensor<2xi32>) <{alignment = 64 : i64}> constant : !fir.array<2xi32>
fir.global internal @_QQro.3xi4.0(dense<[1, 2, 3]> : tensor<3xi32>) <{alignment = 64 : i64}> constant : !fir.array<3xi32>

// The call cannot access a.
// CHECK-LABEL: func.func @_QMmPpositive(
// CHECK-NOT: .tmp.expr_result
// CHECK: fir.store %{{.*}} to %[[A:.*]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[BOX:.*]] = fir.load %[[A]]#0
// CHECK-NEXT: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
// CHECK-NEXT: %[[REF:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK-NEXT: %[[RES:.*]] = fir.call @_QMmPindep(
// CHECK-NEXT: fir.save_result %[[RES]] to %[[REF]](
// CHECK-NOT: hlfir.assign
// CHECK: return

func.func @_QMmPpositive(%arg0: !fir.ref<i32> {fir.bindc_name = "r", fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %true = arith.constant true
  %false = arith.constant false
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c3_i32 = arith.constant 3 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> <{bindc_name = "a", uniq_name = "_QMmFpositiveEa"}>
  %2 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %3 = fir.shape %c0 : (index) -> !fir.shape<1>
  %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %5:2 = hlfir.declare %1 uniq_name("_QMmFpositiveEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %6:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFpositiveEr") fortran_attrs<intent_out> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7:3 = hlfir.associate %c3_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  %8:2 = hlfir.declare %7#0 uniq_name("_QMmFindepEn") fortran_attrs<intent_in> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9 = fir.load %8#0 : !fir.ref<i32>
  %10 = fir.convert %9 : (i32) -> index
  %11 = arith.maxsi %10, %c0 : index
  %12 = fir.shape %11 : (index) -> !fir.shape<1>
  %13 = hlfir.eval_in_mem shape %12 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: !fir.ref<!fir.array<?xi32>>):
    %27 = fir.call @_QMmPindep(%7#0) fastmath<contract> : (!fir.ref<i32>) -> !fir.array<?xi32>
    fir.save_result %27 to %arg1(%12) : !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>, !fir.shape<1>
  }
  %14 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %15 = fir.box_addr %14 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %16 = fir.convert %15 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %17 = arith.cmpi ne, %16, %c0_i64 : i64
  %18:2 = fir.if %17 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %27:3 = fir.box_dims %14, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %28 = arith.cmpi ne, %27#1, %11 : index
    %29 = arith.select %28, %28, %false : i1
    %30 = fir.if %29 -> (!fir.heap<!fir.array<?xi32>>) {
      %31 = fir.allocmem !fir.array<?xi32>, %11 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %31 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %15 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %29, %30 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %27 = fir.allocmem !fir.array<?xi32>, %11 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %27 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %18#0 {
    fir.if %17 {
      fir.freemem %15 : !fir.heap<!fir.array<?xi32>>
    }
    %27 = fir.embox %18#1(%12) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %27 to %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %19 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %13 to %19 : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  hlfir.end_associate %7#1, %7#2 : !fir.ref<i32>, i1
  hlfir.destroy %13 : !hlfir.expr<?xi32>
  return
}

// The call reads a through its argument.
// CHECK-LABEL: func.func @_QMmPvia_argument(
// CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} <{bindc_name = ".tmp.expr_result"
// CHECK: %[[RES:.*]] = fir.call @_QMmPreads_arg(
// CHECK-NEXT: fir.save_result %[[RES]] to
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>

func.func @_QMmPvia_argument(%arg0: !fir.ref<i32> {fir.bindc_name = "r", fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> <{bindc_name = "a", uniq_name = "_QMmFvia_argumentEa"}>
  %2 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %3 = fir.shape %c0 : (index) -> !fir.shape<1>
  %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %5:2 = hlfir.declare %1 uniq_name("_QMmFvia_argumentEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %6:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFvia_argumentEr") fortran_attrs<intent_out> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.address_of(@_QQro.3xi4.0) : !fir.ref<!fir.array<3xi32>>
  %8 = fir.shape %c3 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %7(%8) uniq_name("_QQro.3xi4.0") fortran_attrs<parameter> : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  %10 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %11 = fir.box_addr %10 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %12 = fir.convert %11 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %13 = arith.cmpi ne, %12, %c0_i64 : i64
  %14:2 = fir.if %13 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %41:3 = fir.box_dims %10, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %42 = arith.cmpi ne, %41#1, %c3 : index
    %43 = arith.select %42, %42, %false : i1
    %44 = fir.if %43 -> (!fir.heap<!fir.array<?xi32>>) {
      %45 = fir.allocmem !fir.array<?xi32>, %c3 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %45 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %11 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %43, %44 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %41 = fir.allocmem !fir.array<?xi32>, %c3 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %41 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %14#0 {
    fir.if %13 {
      fir.freemem %11 : !fir.heap<!fir.array<?xi32>>
    }
    %41 = fir.shape_shift %c1, %c3 : (index, index) -> !fir.shapeshift<1>
    %42 = fir.embox %14#1(%41) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %42 to %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %15 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %9#0 to %15 : !fir.ref<!fir.array<3xi32>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  %16 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %17 = fir.rebox %16 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<!fir.array<?xi32>>
  %18:2 = hlfir.declare %17 uniq_name("_QMmFreads_argEx") fortran_attrs<intent_in> : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %19:3 = fir.box_dims %18#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %20 = fir.convert %19#1 : (index) -> i32
  %21 = arith.addi %20, %c1_i32 : i32
  %22 = fir.convert %21 : (i32) -> index
  %23 = arith.maxsi %22, %c0 : index
  %24 = fir.shape %23 : (index) -> !fir.shape<1>
  %25 = hlfir.eval_in_mem shape %24 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: !fir.ref<!fir.array<?xi32>>):
    %41 = fir.call @_QMmPreads_arg(%17) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    fir.save_result %41 to %arg1(%24) : !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>, !fir.shape<1>
  }
  %26 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %27 = fir.box_addr %26 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %28 = fir.convert %27 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %29 = arith.cmpi ne, %28, %c0_i64 : i64
  %30:2 = fir.if %29 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %41:3 = fir.box_dims %26, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %42 = arith.cmpi ne, %41#1, %23 : index
    %43 = arith.select %42, %42, %false : i1
    %44 = fir.if %43 -> (!fir.heap<!fir.array<?xi32>>) {
      %45 = fir.allocmem !fir.array<?xi32>, %23 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %45 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %27 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %43, %44 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %41 = fir.allocmem !fir.array<?xi32>, %23 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %41 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %30#0 {
    fir.if %29 {
      fir.freemem %27 : !fir.heap<!fir.array<?xi32>>
    }
    %41 = fir.embox %30#1(%24) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %41 to %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %31 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %25 to %31 : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  hlfir.destroy %25 : !hlfir.expr<?xi32>
  return
}

// The call reads the module variable ga.
// CHECK-LABEL: func.func @_QMmPvia_module(
// CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} <{bindc_name = ".tmp.expr_result"
// CHECK: %[[RES:.*]] = fir.call @_QMmPreads_module(
// CHECK-NEXT: fir.save_result %[[RES]] to
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>

func.func @_QMmPvia_module(%arg0: !fir.ref<i32> {fir.bindc_name = "r", fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i64 = arith.constant 0 : i64
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c4_i32 = arith.constant 4 : i32
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMmEga) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2:2 = hlfir.declare %1 uniq_name("_QMmEga") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFvia_moduleEr") fortran_attrs<intent_out> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.allocmem !fir.array<?xi32>, %c2 <{alignment = 64 : i64, uniq_name = "_QMmEga.alloc"}> {fir.must_be_heap = true}
  %5 = fir.shape %c2 : (index) -> !fir.shape<1>
  %6 = fir.embox %4(%5) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %6 to %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %7 = fir.address_of(@_QQro.2xi4.2) : !fir.ref<!fir.array<2xi32>>
  %8:2 = hlfir.declare %7(%5) uniq_name("_QQro.2xi4.2") fortran_attrs<parameter> : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  %9 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %10 = fir.box_addr %9 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %11 = fir.convert %10 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %12 = arith.cmpi ne, %11, %c0_i64 : i64
  %13:2 = fir.if %12 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %38:3 = fir.box_dims %9, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %39 = arith.cmpi ne, %38#1, %c2 : index
    %40 = arith.select %39, %39, %false : i1
    %41 = fir.if %40 -> (!fir.heap<!fir.array<?xi32>>) {
      %42 = fir.allocmem !fir.array<?xi32>, %c2 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %42 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %10 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %40, %41 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %38 = fir.allocmem !fir.array<?xi32>, %c2 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %38 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %13#0 {
    fir.if %12 {
      fir.freemem %10 : !fir.heap<!fir.array<?xi32>>
    }
    %38 = fir.shape_shift %c1, %c2 : (index, index) -> !fir.shapeshift<1>
    %39 = fir.embox %13#1(%38) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %39 to %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %14 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %8#0 to %14 : !fir.ref<!fir.array<2xi32>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  %15:3 = hlfir.associate %c4_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  %16:2 = hlfir.declare %15#0 uniq_name("_QMmFreads_moduleEn") fortran_attrs<intent_in> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %17 = fir.load %16#0 : !fir.ref<i32>
  %18 = fir.convert %17 : (i32) -> index
  %19 = arith.maxsi %18, %c0 : index
  %20 = fir.shape %19 : (index) -> !fir.shape<1>
  %21 = hlfir.eval_in_mem shape %20 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: !fir.ref<!fir.array<?xi32>>):
    %38 = fir.call @_QMmPreads_module(%15#0) fastmath<contract> : (!fir.ref<i32>) -> !fir.array<?xi32>
    fir.save_result %38 to %arg1(%20) : !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>, !fir.shape<1>
  }
  %22 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %23 = fir.box_addr %22 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %24 = fir.convert %23 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %25 = arith.cmpi ne, %24, %c0_i64 : i64
  %26:2 = fir.if %25 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %38:3 = fir.box_dims %22, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %39 = arith.cmpi ne, %38#1, %19 : index
    %40 = arith.select %39, %39, %false : i1
    %41 = fir.if %40 -> (!fir.heap<!fir.array<?xi32>>) {
      %42 = fir.allocmem !fir.array<?xi32>, %19 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %42 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %23 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %40, %41 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %38 = fir.allocmem !fir.array<?xi32>, %19 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %38 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %26#0 {
    fir.if %25 {
      fir.freemem %23 : !fir.heap<!fir.array<?xi32>>
    }
    %38 = fir.embox %26#1(%20) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %38 to %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %27 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %21 to %27 : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  hlfir.end_associate %15#1, %15#2 : !fir.ref<i32>, i1
  hlfir.destroy %21 : !hlfir.expr<?xi32>
  return
}

// The call reads the TARGET a through the pointer p.
// CHECK-LABEL: func.func @_QMmPvia_pointer(
// CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} <{bindc_name = ".tmp.expr_result"
// CHECK: %[[RES:.*]] = fir.call @_QMmPreads_arg(
// CHECK-NEXT: fir.save_result %[[RES]] to
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>

func.func @_QMmPvia_pointer(%arg0: !fir.ref<i32> {fir.bindc_name = "r", fir.fortran_attrs = #fir.var_attrs<intent_out>}) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c1_i32 = arith.constant 1 : i32
  %c3 = arith.constant 3 : index
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> <{bindc_name = "a", uniq_name = "_QMmFvia_pointerEa"}> {fir.target}
  %2 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %3 = fir.shape %c0 : (index) -> !fir.shape<1>
  %4 = fir.embox %2(%3) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %4 to %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %5:2 = hlfir.declare %1 uniq_name("_QMmFvia_pointerEa") fortran_attrs<allocatable, target> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %6 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> <{bindc_name = "p", uniq_name = "_QMmFvia_pointerEp"}>
  %7 = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
  %8 = fir.embox %7(%3) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %8 to %6 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %9:2 = hlfir.declare %6 uniq_name("_QMmFvia_pointerEp") fortran_attrs<pointer> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
  %10:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFvia_pointerEr") fortran_attrs<intent_out> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %11 = fir.address_of(@_QQro.3xi4.0) : !fir.ref<!fir.array<3xi32>>
  %12 = fir.shape %c3 : (index) -> !fir.shape<1>
  %13:2 = hlfir.declare %11(%12) uniq_name("_QQro.3xi4.0") fortran_attrs<parameter> : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
  %14 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %15 = fir.box_addr %14 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %16 = fir.convert %15 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %17 = arith.cmpi ne, %16, %c0_i64 : i64
  %18:2 = fir.if %17 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %49:3 = fir.box_dims %14, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %50 = arith.cmpi ne, %49#1, %c3 : index
    %51 = arith.select %50, %50, %false : i1
    %52 = fir.if %51 -> (!fir.heap<!fir.array<?xi32>>) {
      %53 = fir.allocmem !fir.array<?xi32>, %c3 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %53 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %15 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %51, %52 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %49 = fir.allocmem !fir.array<?xi32>, %c3 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %49 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %18#0 {
    fir.if %17 {
      fir.freemem %15 : !fir.heap<!fir.array<?xi32>>
    }
    %49 = fir.shape_shift %c1, %c3 : (index, index) -> !fir.shapeshift<1>
    %50 = fir.embox %18#1(%49) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %50 to %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %19 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %13#0 to %19 : !fir.ref<!fir.array<3xi32>>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  %20 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %21:3 = fir.box_dims %20, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %22 = fir.shift %21#0 : (index) -> !fir.shift<1>
  %23 = fir.rebox %20(%22) : (!fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
  fir.store %23 to %9#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %24 = fir.load %9#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  %25 = fir.rebox %24 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<!fir.array<?xi32>>
  %26:2 = hlfir.declare %25 uniq_name("_QMmFreads_argEx") fortran_attrs<intent_in> : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %27:3 = fir.box_dims %26#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %28 = fir.convert %27#1 : (index) -> i32
  %29 = arith.addi %28, %c1_i32 : i32
  %30 = fir.convert %29 : (i32) -> index
  %31 = arith.maxsi %30, %c0 : index
  %32 = fir.shape %31 : (index) -> !fir.shape<1>
  %33 = hlfir.eval_in_mem shape %32 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: !fir.ref<!fir.array<?xi32>>):
    %49 = fir.call @_QMmPreads_arg(%25) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    fir.save_result %49 to %arg1(%32) : !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>, !fir.shape<1>
  }
  %34 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %35 = fir.box_addr %34 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %36 = fir.convert %35 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %37 = arith.cmpi ne, %36, %c0_i64 : i64
  %38:2 = fir.if %37 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %49:3 = fir.box_dims %34, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %50 = arith.cmpi ne, %49#1, %31 : index
    %51 = arith.select %50, %50, %false : i1
    %52 = fir.if %51 -> (!fir.heap<!fir.array<?xi32>>) {
      %53 = fir.allocmem !fir.array<?xi32>, %31 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %53 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %35 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %51, %52 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %49 = fir.allocmem !fir.array<?xi32>, %31 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %49 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %38#0 {
    fir.if %37 {
      fir.freemem %35 : !fir.heap<!fir.array<?xi32>>
    }
    %49 = fir.embox %38#1(%32) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %49 to %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %39 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %33 to %39 : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  hlfir.destroy %33 : !hlfir.expr<?xi32>
  return
}

// The host-associated a is passed to the call.
// CHECK-LABEL: func.func private @_QMmFvia_hostPinner(
// CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} <{bindc_name = ".tmp.expr_result"
// CHECK: %[[RES:.*]] = fir.call @_QMmPreads_arg(
// CHECK-NEXT: fir.save_result %[[RES]] to
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>

func.func private @_QMmFvia_hostPinner(%arg0: !fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QMmPvia_host, llvm.linkage = #llvm.linkage<internal>} {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>>
  %2 = fir.load %1 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>>
  %3:2 = hlfir.declare %2 uniq_name("_QMmFvia_hostEa") fortran_attrs<allocatable, host_assoc> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %4 = fir.load %3#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %5 = fir.rebox %4 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.box<!fir.array<?xi32>>
  %6:2 = hlfir.declare %5 uniq_name("_QMmFreads_argEx") fortran_attrs<intent_in> : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %7:3 = fir.box_dims %6#0, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %8 = fir.convert %7#1 : (index) -> i32
  %9 = arith.addi %8, %c1_i32 : i32
  %10 = fir.convert %9 : (i32) -> index
  %11 = arith.maxsi %10, %c0 : index
  %12 = fir.shape %11 : (index) -> !fir.shape<1>
  %13 = hlfir.eval_in_mem shape %12 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: !fir.ref<!fir.array<?xi32>>):
    %20 = fir.call @_QMmPreads_arg(%5) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
    fir.save_result %20 to %arg1(%12) : !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>, !fir.shape<1>
  }
  %14 = fir.load %3#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %15 = fir.box_addr %14 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %16 = fir.convert %15 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %17 = arith.cmpi ne, %16, %c0_i64 : i64
  %18:2 = fir.if %17 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %20:3 = fir.box_dims %14, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %21 = arith.cmpi ne, %20#1, %11 : index
    %22 = arith.select %21, %21, %false : i1
    %23 = fir.if %22 -> (!fir.heap<!fir.array<?xi32>>) {
      %24 = fir.allocmem !fir.array<?xi32>, %11 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %24 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %15 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %22, %23 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %20 = fir.allocmem !fir.array<?xi32>, %11 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %20 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %18#0 {
    fir.if %17 {
      fir.freemem %15 : !fir.heap<!fir.array<?xi32>>
    }
    %20 = fir.embox %18#1(%12) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %20 to %3#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %19 = fir.load %3#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %13 to %19 : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  hlfir.destroy %13 : !hlfir.expr<?xi32>
  return
}

// The call may invoke save_recursive, which may access its saved a.
// CHECK-LABEL: func.func @_QMmPsave_recursive(
// CHECK: %[[TMP:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} <{bindc_name = ".tmp.expr_result"
// CHECK: %[[RES:.*]] = fir.call @_QMmPindep(
// CHECK-NEXT: fir.save_result %[[RES]] to
// CHECK: hlfir.assign %{{.*}} to %{{.*}} : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>

func.func @_QMmPsave_recursive(%arg0: !fir.ref<i32> {fir.bindc_name = "r", fir.fortran_attrs = #fir.var_attrs<intent_out>}) attributes {fir.proc_attrs = #fir.proc_attrs<recursive>} {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c3_i32 = arith.constant 3 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMmFsave_recursiveEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2:2 = hlfir.declare %1 uniq_name("_QMmFsave_recursiveEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFsave_recursiveEr") fortran_attrs<intent_out> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:3 = hlfir.associate %c3_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  %5:2 = hlfir.declare %4#0 uniq_name("_QMmFindepEn") fortran_attrs<intent_in> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.load %5#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.maxsi %7, %c0 : index
  %9 = fir.shape %8 : (index) -> !fir.shape<1>
  %10 = hlfir.eval_in_mem shape %9 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: !fir.ref<!fir.array<?xi32>>):
    %20 = fir.call @_QMmPindep(%4#0) fastmath<contract> : (!fir.ref<i32>) -> !fir.array<?xi32>
    fir.save_result %20 to %arg1(%9) : !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>, !fir.shape<1>
  }
  %11 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %12 = fir.box_addr %11 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %13 = fir.convert %12 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %14 = arith.cmpi ne, %13, %c0_i64 : i64
  %15:2 = fir.if %14 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %20:3 = fir.box_dims %11, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %21 = arith.cmpi ne, %20#1, %8 : index
    %22 = arith.select %21, %21, %false : i1
    %23 = fir.if %22 -> (!fir.heap<!fir.array<?xi32>>) {
      %24 = fir.allocmem !fir.array<?xi32>, %8 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %24 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %12 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %22, %23 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %20 = fir.allocmem !fir.array<?xi32>, %8 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %20 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %15#0 {
    fir.if %14 {
      fir.freemem %12 : !fir.heap<!fir.array<?xi32>>
    }
    %20 = fir.embox %15#1(%9) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %20 to %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %16 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %10 to %16 : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  hlfir.end_associate %4#1, %4#2 : !fir.ref<i32>, i1
  hlfir.destroy %10 : !hlfir.expr<?xi32>
  return
}

// The call cannot invoke the NON_RECURSIVE save_non_recursive.
// CHECK-LABEL: func.func @_QMmPsave_non_recursive(
// CHECK-NOT: .tmp.expr_result
// CHECK: fir.store %{{.*}} to %[[A:.*]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK: %[[BOX:.*]] = fir.load %[[A]]#0
// CHECK-NEXT: %[[ADDR:.*]] = fir.box_addr %[[BOX]]
// CHECK-NEXT: %[[REF:.*]] = fir.convert %[[ADDR]] : (!fir.heap<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK-NEXT: %[[RES:.*]] = fir.call @_QMmPindep(
// CHECK-NEXT: fir.save_result %[[RES]] to %[[REF]](
// CHECK-NOT: hlfir.assign
// CHECK: return

func.func @_QMmPsave_non_recursive(%arg0: !fir.ref<i32> {fir.bindc_name = "r", fir.fortran_attrs = #fir.var_attrs<intent_out>}) attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i64 = arith.constant 0 : i64
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c3_i32 = arith.constant 3 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMmFsave_non_recursiveEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2:2 = hlfir.declare %1 uniq_name("_QMmFsave_non_recursiveEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFsave_non_recursiveEr") fortran_attrs<intent_out> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4:3 = hlfir.associate %c3_i32 {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  %5:2 = hlfir.declare %4#0 uniq_name("_QMmFindepEn") fortran_attrs<intent_in> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.load %5#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> index
  %8 = arith.maxsi %7, %c0 : index
  %9 = fir.shape %8 : (index) -> !fir.shape<1>
  %10 = hlfir.eval_in_mem shape %9 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg1: !fir.ref<!fir.array<?xi32>>):
    %20 = fir.call @_QMmPindep(%4#0) fastmath<contract> : (!fir.ref<i32>) -> !fir.array<?xi32>
    fir.save_result %20 to %arg1(%9) : !fir.array<?xi32>, !fir.ref<!fir.array<?xi32>>, !fir.shape<1>
  }
  %11 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %12 = fir.box_addr %11 : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
  %13 = fir.convert %12 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %14 = arith.cmpi ne, %13, %c0_i64 : i64
  %15:2 = fir.if %14 -> (i1, !fir.heap<!fir.array<?xi32>>) {
    %20:3 = fir.box_dims %11, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %21 = arith.cmpi ne, %20#1, %8 : index
    %22 = arith.select %21, %21, %false : i1
    %23 = fir.if %22 -> (!fir.heap<!fir.array<?xi32>>) {
      %24 = fir.allocmem !fir.array<?xi32>, %8 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
      fir.result %24 : !fir.heap<!fir.array<?xi32>>
    } else {
      fir.result %12 : !fir.heap<!fir.array<?xi32>>
    }
    fir.result %22, %23 : i1, !fir.heap<!fir.array<?xi32>>
  } else {
    %20 = fir.allocmem !fir.array<?xi32>, %8 <{alignment = 64 : i64, uniq_name = ".auto.alloc"}>
    fir.result %true, %20 : i1, !fir.heap<!fir.array<?xi32>>
  }
  fir.if %15#0 {
    fir.if %14 {
      fir.freemem %12 : !fir.heap<!fir.array<?xi32>>
    }
    %20 = fir.embox %15#1(%9) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
    fir.store %20 to %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  }
  %16 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %10 to %16 : !hlfir.expr<?xi32>, !fir.box<!fir.heap<!fir.array<?xi32>>>
  hlfir.end_associate %4#1, %4#2 : !fir.ref<i32>, i1
  hlfir.destroy %10 : !hlfir.expr<?xi32>
  return
}

func.func private @_QMmPindep(%arg0: !fir.ref<i32> {fir.bindc_name = "n", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}) -> !fir.array<?xi32>
func.func private @_QMmPreads_arg(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x", fir.fortran_attrs = #fir.var_attrs<intent_in>}) -> !fir.array<?xi32>
func.func private @_QMmPreads_module(%arg0: !fir.ref<i32> {fir.bindc_name = "n", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}) -> !fir.array<?xi32>
func.func private @_QMmPvia_host(!fir.ref<i32>)
