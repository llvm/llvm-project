! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine ss5()
common /com_dummy1/ x
! CHECK: fir.global common @com_dummy1_
interface
    subroutine com_dummy1()
    end subroutine
end interface
! CHECK: func.func private @_QPcom_dummy1()
print *,fun_sub(com_dummy1)
end
