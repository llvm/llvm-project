! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

module m
  integer, private :: cm[*]
contains
  integer function ub(d)
    integer, intent(in) :: d
    ub = ucobound(cm, dim=d)
  end function
end module

program p
  integer :: c[*]
  c = 5
  if (this_image() == 1) call show
contains
  subroutine show
    c = c + 5
  end subroutine
end program

subroutine s
  integer, save :: cs[*]
  call inner
contains
  subroutine inner
    cs = 5
  end subroutine
end subroutine

! CHECK-LABEL: func.func @_QQmain()
! CHECK:       mif.alloc_coarray %{{.*}} {{.*}} uniq_name("_QFEc")
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       fir.call @_QFPshow()
! CHECK:       return

! mif.alloc_coarray must not appear in the body of @_QFPshow
! CHECK-LABEL: func.func private @_QFPshow()
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       hlfir.declare %{{.*}} uniq_name("_QFEc")
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       return

! mif.alloc_coarray must not be duplicated in a subroutine with a SAVE coarray
! CHECK-LABEL: func.func @_QPs()
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       fir.call @_QFsPinner()
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       return

! mif.alloc_coarray must not appear in the internal procedure either
! CHECK-LABEL: func.func private @_QFsPinner()
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       hlfir.declare %{{.*}} uniq_name("_QFsEcs")
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       return

! mif.alloc_coarray is still in __mif_save_coarrays_allocate for modules variables
! CHECK-LABEL: func.func @__mif_save_coarrays_allocate()
! CHECK-DAG:   mif.alloc_coarray %{{.*}} {{.*}} uniq_name("_QMmEcm")
! CHECK-DAG:   mif.alloc_coarray %{{.*}} {{.*}} uniq_name("_QFsEcs")
! CHECK-NOT:   mif.alloc_coarray
! CHECK:       return
