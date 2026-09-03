! Pinned negative tests for -finit-local= exclusions.
!
! Each case uses %flang_fc1 -emit-hlfir -finit-local=0xAA and asserts that
! no synthesized initialization store appears for the excluded variable.
!
! Exclusions verified here:
!   1. Function result variable
!   2. Main-program local (implicit SAVE per Fortran 2018 8.5.16p4)
!   3. Runtime-sized automatic array (unknown extent at compile time)
!   4. CHARACTER array (pending memset infrastructure, PR #159788)
!   5. Derived type with POINTER component (IsAllocatableOrPointer on component)
!   6. host,device CUDA local (allocated via cuf.alloc, not fir.alloca)
!      -- see finit-local-cuda.cuf for the CUDA cases
!   7. PowerPC vector locals (direct or as a derived-type component)
!      -- see finit-local-ppc-vector-llvm.f90 for the vector cases

! RUN: %flang_fc1 -emit-hlfir -finit-local=0xAA %s -o - | FileCheck %s

! ---------------------------------------------------------------------------
! 1. Function result variable -- must NOT be pre-initialized.
!    The function body owns the result; a synthesized store would overwrite
!    a value already assigned before the body's first user store.
! ---------------------------------------------------------------------------
function test_func_result() result(x)
  integer :: x
  x = 42
end function
! CHECK-LABEL: func.func @_QPtest_func_result
! CHECK:       hlfir.declare {{.*}} "_QFtest_func_resultEx"
! CHECK-NOT:   fir.store
! CHECK:       hlfir.assign
! CHECK:       return

! ---------------------------------------------------------------------------
! 2. Main-program local -- implicit SAVE (Fortran 2018 8.5.16p4).
!    IsSaved() misses the implicit case; the scope-kind guard catches it.
! ---------------------------------------------------------------------------
program test_main_prog
  integer :: n
  n = 1
end program
! CHECK-LABEL: func.func @_QQmain
! CHECK:       hlfir.declare {{.*}} "_QFEn"
! CHECK-NOT:   fir.store
! CHECK:       hlfir.assign
! CHECK:       return

! ---------------------------------------------------------------------------
! 3. Runtime-sized automatic array -- extent is unknown at compile time.
!    The array base is a fir.box; no compile-time trip count is available.
! ---------------------------------------------------------------------------
subroutine test_runtime_array(res, n)
  integer, intent(in) :: n
  integer :: x(n)
  integer :: res
  res = x(1)
end subroutine
! CHECK-LABEL: func.func @_QPtest_runtime_array
! CHECK:       hlfir.declare {{.*}} "_QFtest_runtime_arrayEx"
! CHECK-NOT:   fir.do_loop
! CHECK:       return

! ---------------------------------------------------------------------------
! 4. CHARACTER array -- pending memset infrastructure (PR #159788).
!    Static CHARACTER scalars are initialized; arrays are silently skipped.
! ---------------------------------------------------------------------------
subroutine test_char_array(res)
  character(4) :: x(3)
  character(4) :: res
  res = x(1)
end subroutine
! CHECK-LABEL: func.func @_QPtest_char_array
! CHECK:       hlfir.declare {{.*}} "_QFtest_char_arrayEx"
! CHECK-NOT:   fir.do_loop
! CHECK:       return

! ---------------------------------------------------------------------------
! 5. Derived type with POINTER component -- the component descriptor must
!    not be byte-stomped; IsAllocatableOrPointer on the record catches it.
! ---------------------------------------------------------------------------
subroutine test_pointer_comp(res)
  type t
    integer, pointer :: p
  end type
  type(t) :: x
  integer :: res
  res = 0
end subroutine
! CHECK-LABEL: func.func @_QPtest_pointer_comp
! CHECK:       hlfir.declare {{.*}} "_QFtest_pointer_compEx"
! CHECK-NOT:   fir.do_loop
! CHECK:       return
