! Test that module variables with an initializer are only defined once,
! except for compiler generated derived type descriptor that should be
! always fully defined as linkonce_odr by the compilation units defining or
! using them.
! Test that this holds true in contexts with namelist members that are special
! because the symbol on the use site are not symbols with semantics::UseDetails,
! but directly the symbols from the module scope.


! RUN: split-file %s %t
! RUN: bbc -emit-fir %t/definition-a.f90 -o - | FileCheck %s --check-prefix=CHECK-A-DEF
! RUN: bbc -emit-fir %t/definition-b.f90 -o - | FileCheck %s --check-prefix=CHECK-B-DEF
! RUN: bbc -emit-fir %t/use.f90 -o - | FileCheck %s --check-prefix=CHECK-USE



!--- definition-a.f90

! Test definition of `atype` derived type descriptor as `linkonce_odr`
module define_a
  type atype
    real :: x
  end type
end module

! CHECK-A-DEF: fir.global linkonce_odr @_QMdefine_aE.dt.atype constant target : !fir.type<{{.*}}> {
! CHECK-A-DEF: fir.has_value
! CHECK-A-DEF: }

!--- definition-b.f90

! Test define_b `i` is defined here.
! Also test that the derived type descriptor of types defined here (`btype`) and used
! here (`atype`) are fully defined here as linkonce_odr.
module define_b
  use :: define_a
  type btype
    type(atype) :: atype
  end type
  integer :: i = 42
  namelist /some_namelist/ i
end module

! CHECK-B-DEF: fir.global @_QMdefine_bEi : i32 {
! CHECK-B-DEF: fir.has_value %{{.*}} : i32
! CHECK-B-DEF: }

! CHECK-B-DEF: fir.global linkonce_odr @_QMdefine_bE.dt.btype constant target : !fir.type<{{.*}}> {
! CHECK-B-DEF: fir.has_value
! CHECK-B-DEF: }

! CHECK-B-DEF: fir.global linkonce_odr @_QMdefine_aE.dt.atype constant : !fir.type<{{.*}}> {
! CHECK-B-DEF: fir.has_value
! CHECK-B-DEF: }



!--- use.f90

! Test  define_b `i` is declared but not defined here and that derived types
! descriptors are fully defined as linkonce_odr here.
subroutine foo()
  use :: define_b
  type(btype) :: somet
  print *, somet
  write(*, some_namelist)
end subroutine
! CHECK-USE: fir.global @_QMdefine_bEi : i32{{$}}
! CHECK-USE-NOT: fir.has_value %{{.*}} : i32

! CHECK-USE: fir.global linkonce_odr @_QMdefine_aE.dt.atype constant : !fir.type<{{.*}}> {
! CHECK-USE: fir.has_value
! CHECK-USE: }

! CHECK-USE: fir.global linkonce_odr @_QMdefine_bE.dt.btype constant : !fir.type<{{.*}}> {
! CHECK-USE: fir.has_value
! CHECK-USE: }

