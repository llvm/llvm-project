! REQUIRES: module-independent
! RUN: split-file %s %t
! RUN: %flang_fc1 -fopenmp -emit-hlfir -module-dir %t %t/owner.f90 -o %t.owner
! RUN: FileCheck %s --check-prefix=IR < %t.owner
! RUN: FileCheck %s --check-prefix=UNUSED < %t.owner
! RUN: FileCheck %s --check-prefix=CALLS --implicit-check-not='fir.alloca tuple' < %t.owner
! CALLS-COUNT-2: fir.global internal @_QFE{{(first|second)}} : i32
! IR-LABEL: func.func private @_QFPworker()
! IR-DAG: fir.address_of(@_QFEalias)
! IR-DAG: fir.address_of(@_QFEx)
! IR: omp.parallel
! IR: return
! IR-LABEL: func.func private @_QFPother()
! IR-DAG: %[[MODULE:[^ ]+]] = fir.address_of(@_QMneededEa)
! IR-DAG: %[[DEEP_ADDR:[^ ]+]] = fir.address_of(@_QMtail0Ecaptured)
! IR-DAG: %[[DEEP:[^: ]+]]:2 = hlfir.declare %[[DEEP_ADDR]] {{.*}}uniq_name("_QMtail0Ecaptured")
! IR-DAG: %[[COMMON:[^ ]+]] = fir.address_of(@block_)
! IR-DAG: %[[USED:[^ ]+]]:2 = hlfir.declare {{.*}} storage(%[[MODULE]][0]) uniq_name("_QMneededEb")
! IR-DAG: %[[SHARED:[^ ]+]]:2 = hlfir.declare {{.*}} storage(%[[COMMON]][0]) uniq_name("_QFEshared")
! IR: omp.parallel
! IR-DAG: fir.load %[[USED]]#0 :
! IR-DAG: fir.load %[[SHARED]]#0 :
! IR-DAG: fir.load %[[DEEP]]#0 :
! IR: return
! UNUSED-LABEL: func.func private @_QFPworker()
! UNUSED-NOT: fir.address_of(@_QM{{(neededEc|tail0Ev)}}
! UNUSED-NOT: hlfir.declare {{.*}}uniq_name("_QMneededEa")
! UNUSED: return
! UNUSED-LABEL: func.func private @_QFPother()
!--- owner.f90
! Internal procedures bind used globals and shared EQUIVALENCE/COMMON storage.
module needed
  integer :: a,b,c,d; equivalence(a,b),(c,d)
contains
integer function id(n); integer :: n; id=n; end function
end module
module tail0; integer :: v,captured; end module; module tail1; use tail0, only: v,captured; end module
module tail2; use tail1, only: v,captured; end module; module tail3; use tail2, only: v,captured; end module
program p
  use needed, only: unused=>a,used=>b,id
  use tail3, only: a1=>v,a2=>v,a3=>v,a4=>v,deep=>captured
  integer,save :: backing(2),alias; equivalence(backing(1),alias)
  integer :: x,first,second,shared; common /block/ shared
  alias=3; x=7; used=11; shared=13; first=1; second=2; deep=17; call worker(); call other()
contains
  subroutine worker()
!$omp parallel num_threads(1)
    print *,x,alias,id(first),id(second),id(first)
!$omp end parallel
  end subroutine; subroutine other()
!$omp parallel num_threads(1)
    print *,x,alias,used,shared,deep
!$omp end parallel
  end subroutine; end program
