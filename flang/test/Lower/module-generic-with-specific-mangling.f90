! RUN: split-file %s %t
! RUN: bbc -emit-fir %t/mangling_mod_a.f90 -o - | FileCheck %s --check-prefix=FIR
! RUN: bbc -emit-fir %t/mangling_mod_b.f90 -o - | FileCheck %s --check-prefix=MANGLE
! RUN: bbc -emit-fir %t/mangling_mod_c.f90 -o - | FileCheck %s --check-prefix=MANGLE
! RUN: bbc -emit-fir %t/mangling_mod_d.f90 -o - | FileCheck %s --check-prefix=MANGLE

! FIR: module
! MANGLE: func.func private @_QPmy_sub(!fir.ref<i32>)

!--- mangling_mod_a.f90
module mangling_mod_a
  interface
    subroutine my_sub(a)
      integer :: a
    end subroutine my_sub
  end interface

  ! Generic interface
  interface my_sub
      procedure :: my_sub
  end interface
  contains
end module mangling_mod_a

!--- mangling_mod_b.f90
module mangling_mod_b
  use mangling_mod_a

  contains
    subroutine my_sub2(a)
      integer :: a
      call my_sub(a)
    end subroutine my_sub2

end module mangling_mod_b

!--- mangling_mod_c.f90
module mangling_mod_c
  use mangling_mod_b

  contains
    subroutine my_sub3(a)
      integer :: a

      call my_sub(a)
    end subroutine my_sub3
end module mangling_mod_c

!--- mangling_mod_d.f90
module mangling_mod_d
  use mangling_mod_b
  use mangling_mod_c

  contains
    subroutine my_sub4(a)
      integer :: a

      call my_sub(a)
    end subroutine my_sub4
end module mangling_mod_d
