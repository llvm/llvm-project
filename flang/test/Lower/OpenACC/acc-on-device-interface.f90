! An on_device() function that is only declared, by an interface body or an
! EXTERNAL statement, lowers to cuf.on_device. A definition in the same file
! keeps the call.

! RUN: split-file %s %t
! RUN: bbc -fopenacc -emit-hlfir %t/declared.f90 -o - | FileCheck %s --check-prefix=DECL
! RUN: bbc -emit-hlfir %t/declared.f90 -o - | FileCheck %s --check-prefix=NOACC
! RUN: bbc -fopenacc -emit-hlfir %t/defined.f90 -o - | FileCheck %s --check-prefix=DEF

//--- declared.f90
subroutine use_interface(r)
  interface
    logical function on_device()
      !$acc routine seq
    end function
  end interface
  logical :: r
  r = on_device()
end subroutine

subroutine use_external(r)
  logical, external :: on_device
  logical :: r
  r = on_device()
end subroutine

! DECL-LABEL: func.func @_QPuse_interface
! DECL: cuf.on_device : i1
! DECL-NOT: fir.call @_QPon_device

! DECL-LABEL: func.func @_QPuse_external
! DECL: cuf.on_device : i1
! DECL-NOT: fir.call @_QPon_device

! NOACC-LABEL: func.func @_QPuse_interface
! NOACC: fir.call @_QPon_device()
! NOACC-NOT: cuf.on_device

//--- defined.f90
logical function on_device()
  on_device = .false.
end function

subroutine use_defined(r)
  logical, external :: on_device
  logical :: r
  r = on_device()
end subroutine

! DEF-LABEL: func.func @_QPuse_defined
! DEF: fir.call @_QPon_device()
! DEF-NOT: cuf.on_device
