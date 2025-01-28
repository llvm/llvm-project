! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

program main
!CHECK-LABEL: MainProgram scope: main
  implicit none

  type ty
     integer :: x
  end type ty
  !$omp declare mapper(mymapper : ty :: mapped) map(mapped, mapped%x)
  !$omp declare mapper(ty :: maptwo) map(maptwo, maptwo%x)

!! Note, symbols come out in their respective scope, but not in declaration order.
!CHECK: default: Misc ConstructName
!CHECK: mymapper: Misc ConstructName
!CHECK: ty: DerivedType components: x
!CHECK: DerivedType scope: ty
!CHECK: OtherConstruct scope:
!CHECK: mapped (OmpMapToFrom) {{.*}} ObjectEntity type: TYPE(ty)
!CHECK: OtherConstruct scope:  
!CHECK: maptwo (OmpMapToFrom) {{.*}} ObjectEntity type: TYPE(ty)
  
end program main

