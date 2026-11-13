! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

module m_same_name
  character(:), pointer :: deferred_ptr
  character(:), allocatable :: deferred_alloc
  integer, pointer :: int_ptr
end module

! CHECK-LABEL: func.func @_QPdeferred_ptr
! CHECK-NOT: fir.address_of
! CHECK: return
subroutine deferred_ptr()
  use m_same_name, only: deferred_ptr
end subroutine

! CHECK-LABEL: func.func @_QPdeferred_alloc
! CHECK-NOT: fir.address_of
! CHECK: return
subroutine deferred_alloc()
  use m_same_name, only: deferred_alloc
end subroutine

! CHECK-LABEL: func.func @_QPint_ptr
! CHECK-NOT: fir.address_of
! CHECK: return
subroutine int_ptr()
  use m_same_name, only: int_ptr
end subroutine
