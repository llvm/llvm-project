! RUN: %flang_fc1 -fdebug-dump-symbols -fopenmp -fopenmp-version=50 %s | FileCheck %s

module mm
  implicit none
  type logicalwrapper
     logical b
  end type logicalwrapper

contains
!CHECK-LABEL: Subprogram scope: func
  function func(x, n)
    logical func
    integer :: n
    type(logicalwrapper) ::  x(n)
    type(logicalwrapper) :: res
    integer :: i
    !$omp declare reduction(.AND.:type(logicalwrapper):omp_out%b=omp_out%b .AND. omp_in%b) initializer(omp_priv%b=.true.)
!CHECK: op.AND: UserReductionDetails TYPE(logicalwrapper)
!CHECK OtherConstruct scope
!CHECK: omp_in size=4 offset=0: ObjectEntity type: TYPE(logicalwrapper)
!CHECK: omp_orig size=4 offset=4: ObjectEntity type: TYPE(logicalwrapper)
!CHECK: omp_out size=4 offset=8: ObjectEntity type: TYPE(logicalwrapper)
!CHECK: omp_priv size=4 offset=12: ObjectEntity type: TYPE(logicalwrapper)
  
    !$omp simd reduction(.AND.:res)
    do i=1,n
       res%b=res%b .and. x(i)%b
    enddo
    
    func=res%b
  end function func
end module mm
