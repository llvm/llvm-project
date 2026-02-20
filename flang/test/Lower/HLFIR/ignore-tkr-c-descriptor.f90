! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! Test that ignore_tkr(c) avoids descriptor copies (rebox/embox) for dummy arguments.

module m_ignore_tkr_c
  interface
    subroutine pass_array_ptr(a)
      !dir$ ignore_tkr(cp) a
      real, pointer :: a(:)
    end subroutine
    subroutine pass_array_val(a)
      !dir$ ignore_tkr(c) a
      real :: a(:)
    end subroutine
  end interface
contains
  ! CHECK-LABEL: func.func @_QMm_ignore_tkr_cPs1(
  ! CHECK-SAME: %[[ARR:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  subroutine s1(arr)
    real, allocatable :: arr(:)
    ! CHECK: %[[BOX_REF:.*]]:2 = hlfir.declare %[[ARR]]
    ! CHECK: %[[CONV:.*]] = fir.convert %[[BOX_REF]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: fir.call @_QPpass_array_ptr(%[[CONV]]) {{.*}} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()
    call pass_array_ptr(arr)
  end subroutine

  ! CHECK-LABEL: func.func @_QMm_ignore_tkr_cPs2(
  ! CHECK-SAME: %[[ARR:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  subroutine s2(arr)
    real, allocatable :: arr(:)
    ! CHECK: %[[BOX_REF:.*]]:2 = hlfir.declare %[[ARR]]
    ! CHECK-NOT: fir.load %[[BOX_REF]]#0
    ! CHECK: %[[ADDR:.*]] = fir.address_of(@_QPpass_array_val)
    ! CHECK: %[[CAST:.*]] = fir.convert %[[ADDR]]
    ! CHECK: fir.call %[[CAST]](%[[BOX_REF]]#0) {{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
    call pass_array_val(arr)
  end subroutine

  ! CHECK-LABEL: func.func @_QMm_ignore_tkr_cPs3(
  ! CHECK-SAME: %[[ARR:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  subroutine s3(arr)
    real, allocatable :: arr(:)
    procedure(pass_array_val), pointer :: p
    p => pass_array_val
    ! CHECK: %[[BOX_REF:.*]]:2 = hlfir.declare %[[ARR]]
    ! CHECK: %[[P_REF:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMm_ignore_tkr_cFs3Ep"}
    ! CHECK: %[[ADDR:.*]] = fir.load %[[P_REF]]#0 : !fir.ref<!fir.boxproc<(!fir.box<!fir.array<?xf32>>) -> ()>>
    ! CHECK: %[[FUNC_ADDR:.*]] = fir.box_addr %[[ADDR]] : (!fir.boxproc<(!fir.box<!fir.array<?xf32>>) -> ()>) -> ((!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ())
    ! CHECK: fir.call %[[FUNC_ADDR]](%[[BOX_REF]]#0) {{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> ()
    call p(arr)
  end subroutine

  ! CHECK: func.func private @_QPpass_array_ptr(!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
  ! CHECK: func.func private @_QPpass_array_val(!fir.box<!fir.array<?xf32>>)
end module
