! Test that module data access are lowered correctly in the different
! procedure contexts.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module parent
  integer :: i
contains
! Test simple access to the module data
! CHECK-LABEL: func @_QMparentPtest1
subroutine test1()
  ! CHECK: fir.address_of(@_QMparentEi) : !fir.ref<i32>
  print *, i
end subroutine

! Test access to the module data inside an internal procedure where the
! host is defined inside the module.
subroutine test2()
  call test2internal()
  contains
  ! CHECK-LABEL: func private @_QMparentFtest2Ptest2internal()
  subroutine test2internal()
    ! CHECK: fir.address_of(@_QMparentEi) : !fir.ref<i32>
    print *, i
  end subroutine
end subroutine
end module

! Test access to the module data inside an internal procedure where the
! host is using the module.
subroutine test3()
  use parent
  call test3internal()
  contains
  ! CHECK-LABEL: func private @_QFtest3Ptest3internal()
  subroutine test3internal()
    ! CHECK: fir.address_of(@_QMparentEi) : !fir.ref<i32>
    print *, i
  end subroutine
end subroutine
