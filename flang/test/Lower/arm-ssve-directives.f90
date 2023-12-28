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

!dir$ arm_new_za
subroutine sub4
end subroutine sub4
! CHECK-LABEL: func.func @_QPsub4()
! CHECK-SAME:      attributes {arm_new_za}

!dir$ arm_shared_za
subroutine sub5
end subroutine sub5
! CHECK-LABEL: func.func @_QPsub5()
! CHECK-SAME:      attributes {arm_shared_za}

!dir$ arm_preserves_za
subroutine sub6
end subroutine sub6
! CHECK-LABEL: func.func @_QPsub6()
! CHECK-SAME:      attributes {arm_preserves_za}

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

!dir$ arm_new_za
subroutine msub4
end subroutine msub4
! CHECK-LABEL: func.func @_QMmPmsub4()
! CHECK-SAME:      attributes {arm_new_za}

!dir$ arm_shared_za
subroutine msub5
end subroutine msub5
! CHECK-LABEL: func.func @_QMmPmsub5()
! CHECK-SAME:      attributes {arm_shared_za}

!dir$ arm_preserves_za
subroutine msub6
end subroutine msub6
! CHECK-LABEL: func.func @_QMmPmsub6()
! CHECK-SAME:      attributes {arm_preserves_za}

end module
