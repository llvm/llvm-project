! RUN: bbc -emit-hlfir %s -o - 2>&1 | FileCheck %s

! check we don't warn about these attributes
! CHECK-NOT: warning: Compiler directive was ignored

! check we create the right fuction attributes

!dir$ arm_streaming
subroutine sub
end subroutine sub
! CHECK-LABEL: func.func @_QPsub()
! CHECK-SAME:      attributes {arm_streaming}

!dir$ arm_locally_streaming
subroutine sub2
end subroutine sub2
! CHECK-LABEL: func.func @_QPsub2()
! CHECK-SAME:      attributes {arm_locally_streaming}

!dir$ arm_streaming_compatible
subroutine sub3
end subroutine sub3
! CHECK-LABEL: func.func @_QPsub3()
! CHECK-SAME:      attributes {arm_streaming_compatible}

module m
contains

!dir$ arm_streaming
subroutine msub
end subroutine msub
! CHECK-LABEL: func.func @_QMmPmsub()
! CHECK-SAME:      attributes {arm_streaming}

!dir$ arm_locally_streaming
subroutine msub2
end subroutine msub2
! CHECK-LABEL: func.func @_QMmPmsub2()
! CHECK-SAME:      attributes {arm_locally_streaming}

!dir$ arm_streaming_compatible
subroutine msub3
end subroutine msub3
! CHECK-LABEL: func.func @_QMmPmsub3()
! CHECK-SAME:      attributes {arm_streaming_compatible}
end module
