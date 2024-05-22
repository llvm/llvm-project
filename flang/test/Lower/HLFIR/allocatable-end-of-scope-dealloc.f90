! Test automatic deallocation of local allocatables as described in
! Fortran 2018 standard 9.7.3.2 point 2. and 3.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s
module dtypedef
  type must_finalize
    integer :: i
    contains
      final :: finalize
  end type
  type contain_must_finalize
    type(must_finalize) :: a
  end type
  interface
    subroutine finalize(a)
      import :: must_finalize
      type(must_finalize), intent(inout) :: a
    end subroutine
  end interface
  real, allocatable :: x
end module

subroutine simple()
  real, allocatable :: x
  allocate(x)
  call bar()
end subroutine
! CHECK-LABEL:   func.func @_QPsimple() {
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}"_QFsimpleEx"
! CHECK:  fir.call @_QPbar
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:  %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<f32>) -> i64
! CHECK:  %[[VAL_9:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_9]] : i64
! CHECK:  fir.if %[[VAL_10]] {
! CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:    %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:    fir.freemem %[[VAL_12]] : !fir.heap<f32>
! CHECK:    %[[VAL_13:.*]] = fir.zero_bits !fir.heap<f32>
! CHECK:    %[[VAL_14:.*]] = fir.embox %[[VAL_13]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:    fir.store %[[VAL_14]] to %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:  }

subroutine multiple_return(cdt)
  real, allocatable :: x
  logical :: cdt
  allocate(x)
  if (cdt) return
  call bar()
end subroutine
! CHECK-LABEL:   func.func @_QPmultiple_return(
! CHECK:  cf.cond_br %{{.*}}, ^bb1, ^bb2
! CHECK: ^bb1:
! CHECK-NOT: fir.freemem
! CHECK:  cf.br ^bb3
! CHECK: ^bb2:
! CHECK:  fir.call @_QPbar
! CHECK:  cf.br ^bb3
! CHECK: ^bb3:
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.freemem
! CHECK:  }
! CHECK:  return

subroutine derived()
  use dtypedef, only : must_finalize
  type(must_finalize), allocatable :: x
  allocate(x)
  call bar()
end subroutine
! CHECK-LABEL:   func.func @_QPderived() {
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}"_QFderivedEx"
! CHECK:  fir.call @_QPbar
! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMdtypedefTmust_finalize{i:i32}>>>>
! CHECK:  %[[VAL_12:.*]] = fir.box_addr %[[VAL_11]] : (!fir.box<!fir.heap<!fir.type<_QMdtypedefTmust_finalize{i:i32}>>>) -> !fir.heap<!fir.type<_QMdtypedefTmust_finalize{i:i32}>>
! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (!fir.heap<!fir.type<_QMdtypedefTmust_finalize{i:i32}>>) -> i64
! CHECK:  %[[VAL_14:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_15:.*]] = arith.cmpi ne, %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:  fir.if %[[VAL_15]] {
! CHECK:    %[[VAL_16:.*]] = arith.constant false
! CHECK:    %[[VAL_17:.*]] = fir.absent !fir.box<none>
! CHECK:    %[[VAL_20:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QMdtypedefTmust_finalize{i:i32}>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:    %[[VAL_22:.*]] = fir.call @_FortranAAllocatableDeallocate(%[[VAL_20]], %[[VAL_16]], %[[VAL_17]], %{{.*}}, %{{.*}})
! CHECK:  }

subroutine derived2()
  use dtypedef, only : contain_must_finalize
  type(contain_must_finalize), allocatable :: x
  allocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPderived2(
! CHECK: fir.if {{.*}} {
! CHECK:   fir.call @_FortranAAllocatableDeallocate
! CHECK: }

subroutine simple_block()
  block
    real, allocatable :: x
    allocate(x)
  call bar()
  end block
  call bar_after_block()
end subroutine
! CHECK-LABEL:   func.func @_QPsimple_block(
! CHECK:  fir.call @_QPbar
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.freemem
! CHECK:  }
! CHECK:  fir.call @_QPbar_after_block

subroutine mutiple_return_block(cdt)
  logical :: cdt
  block
    real, allocatable :: x
    allocate(x)
    if (cdt) return
    call bar()
  end block
  call bar_after_block()
end subroutine
! CHECK-LABEL:   func.func @_QPmutiple_return_block(
! CHECK:  cf.cond_br %{{.*}}, ^bb1, ^bb2
! CHECK: ^bb1:
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.freemem
! CHECK:  }
! CHECK:  cf.br ^bb3
! CHECK: ^bb2:
! CHECK:  fir.call @_QPbar
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.freemem
! CHECK:  }
! CHECK:  fir.call @_QPbar_after_block
! CHECK:  cf.br ^bb3
! CHECK: ^bb3:
! CHECK:  return


subroutine derived_block()
  use dtypedef, only : must_finalize
  block
    type(must_finalize), allocatable :: x
    allocate(x)
    call bar()
  end block
  call bar_after_block()
end subroutine
! CHECK-LABEL:   func.func @_QPderived_block(
! CHECK:  fir.call @_QPbar
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.call @_FortranAAllocatableDeallocate
! CHECK:  }
! CHECK:  fir.call @_QPbar_after_block

subroutine derived_block2()
  use dtypedef, only : contain_must_finalize
  call bar()
  block
    type(contain_must_finalize), allocatable :: x
    allocate(x)
  end block
  call bar_after_block()
end subroutine
! CHECK-LABEL:   func.func @_QPderived_block2(
! CHECK:  fir.call @_QPbar
! CHECK:  fir.if {{.*}} {
! CHECK:    fir.call @_FortranAAllocatableDeallocate
! CHECK:  }
! CHECK:  fir.call @_QPbar_after_block

subroutine no_dealloc_saved()
  real, allocatable, save :: x
  allocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPno_dealloc_save
! CHECK-NOT: freemem
! CHECK-NOT: Deallocate
! CHECK: return

subroutine no_dealloc_block_saved()
  block
    real, allocatable, save :: x
    allocate(x)
  end block
end subroutine
! CHECK-LABEL:   func.func @_QPno_dealloc_block_saved
! CHECK-NOT: freemem
! CHECK-NOT: Deallocate
! CHECK: return

function no_dealloc_result() result(x)
  real, allocatable :: x
  allocate(x)
end function
! CHECK-LABEL:   func.func @_QPno_dealloc_result
! CHECK-NOT: freemem
! CHECK-NOT: Deallocate
! CHECK: return

subroutine no_dealloc_dummy(x)
  real, allocatable :: x
  allocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPno_dealloc_dummy
! CHECK-NOT: freemem
! CHECK-NOT: Deallocate
! CHECK: return

subroutine no_dealloc_module_var()
  use dtypedef, only : x
  allocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPno_dealloc_module_var
! CHECK-NOT: freemem
! CHECK-NOT: Deallocate
! CHECK: return

subroutine no_dealloc_host_assoc()
  real, allocatable :: x
  call internal()
contains
  subroutine internal()
    allocate(x)
  end subroutine
end subroutine
! CHECK-LABEL:   func.func @_QFno_dealloc_host_assocPinternal
! CHECK-NOT: freemem
! CHECK-NOT: Deallocate
! CHECK: return

subroutine no_dealloc_pointer(x)
  real, pointer :: x
  allocate(x)
end subroutine
! CHECK-LABEL:   func.func @_QPno_dealloc_pointer
! CHECK-NOT: freemem
! CHECK-NOT: Deallocate
! CHECK: return
