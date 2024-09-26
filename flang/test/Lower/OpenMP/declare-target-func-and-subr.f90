!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s --check-prefixes ALL,HOST
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-device %s -o - | FileCheck %s --check-prefixes ALL,DEVICE

! Check specification valid forms of declare target with functions 
! utilising device_type and to clauses as well as the default 
! zero clause declare target

! DEVICE-LABEL: func.func @_QPfunc_t_device()
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}
FUNCTION FUNC_T_DEVICE() RESULT(I)
!$omp declare target to(FUNC_T_DEVICE) device_type(nohost)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_T_DEVICE

! DEVICE-LABEL: func.func @_QPfunc_enter_device()
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>{{.*}}
FUNCTION FUNC_ENTER_DEVICE() RESULT(I)
!$omp declare target enter(FUNC_ENTER_DEVICE) device_type(nohost)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_ENTER_DEVICE

! HOST-LABEL: func.func @_QPfunc_t_host()
! HOST-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>{{.*}}
FUNCTION FUNC_T_HOST() RESULT(I)
!$omp declare target to(FUNC_T_HOST) device_type(host)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_T_HOST

! HOST-LABEL: func.func @_QPfunc_enter_host()
! HOST-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>{{.*}}
FUNCTION FUNC_ENTER_HOST() RESULT(I)
!$omp declare target enter(FUNC_ENTER_HOST) device_type(host)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_ENTER_HOST

! ALL-LABEL: func.func @_QPfunc_t_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
FUNCTION FUNC_T_ANY() RESULT(I)
!$omp declare target to(FUNC_T_ANY) device_type(any)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_T_ANY

! ALL-LABEL: func.func @_QPfunc_enter_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>{{.*}}
FUNCTION FUNC_ENTER_ANY() RESULT(I)
!$omp declare target enter(FUNC_ENTER_ANY) device_type(any)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_ENTER_ANY

! ALL-LABEL: func.func @_QPfunc_default_t_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
FUNCTION FUNC_DEFAULT_T_ANY() RESULT(I)
!$omp declare target to(FUNC_DEFAULT_T_ANY)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_DEFAULT_T_ANY

! ALL-LABEL: func.func @_QPfunc_default_enter_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>{{.*}}
FUNCTION FUNC_DEFAULT_ENTER_ANY() RESULT(I)
!$omp declare target enter(FUNC_DEFAULT_ENTER_ANY)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_DEFAULT_ENTER_ANY

! ALL-LABEL: func.func @_QPfunc_default_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
FUNCTION FUNC_DEFAULT_ANY() RESULT(I)
!$omp declare target
    INTEGER :: I
    I = 1
END FUNCTION FUNC_DEFAULT_ANY

! ALL-LABEL: func.func @_QPfunc_default_extendedlist()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
FUNCTION FUNC_DEFAULT_EXTENDEDLIST() RESULT(I)
!$omp declare target(FUNC_DEFAULT_EXTENDEDLIST)
    INTEGER :: I
    I = 1
END FUNCTION FUNC_DEFAULT_EXTENDEDLIST

!! -----

! Check specification valid forms of declare target with subroutines 
! utilising device_type and to clauses as well as the default 
! zero clause declare target

! DEVICE-LABEL: func.func @_QPsubr_t_device()
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}
SUBROUTINE SUBR_T_DEVICE()
!$omp declare target to(SUBR_T_DEVICE) device_type(nohost)
END

! DEVICE-LABEL: func.func @_QPsubr_enter_device()
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>{{.*}}
SUBROUTINE SUBR_ENTER_DEVICE()
!$omp declare target enter(SUBR_ENTER_DEVICE) device_type(nohost)
END

! HOST-LABEL: func.func @_QPsubr_t_host()
! HOST-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>{{.*}}
SUBROUTINE SUBR_T_HOST()
!$omp declare target to(SUBR_T_HOST) device_type(host)
END

! HOST-LABEL: func.func @_QPsubr_enter_host()
! HOST-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (enter)>{{.*}}
SUBROUTINE SUBR_ENTER_HOST()
!$omp declare target enter(SUBR_ENTER_HOST) device_type(host)
END

! ALL-LABEL: func.func @_QPsubr_t_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
SUBROUTINE SUBR_T_ANY()
!$omp declare target to(SUBR_T_ANY) device_type(any)
END

! ALL-LABEL: func.func @_QPsubr_enter_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>{{.*}}
SUBROUTINE SUBR_ENTER_ANY()
!$omp declare target enter(SUBR_ENTER_ANY) device_type(any)
END

! ALL-LABEL: func.func @_QPsubr_default_t_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
SUBROUTINE SUBR_DEFAULT_T_ANY()
!$omp declare target to(SUBR_DEFAULT_T_ANY)
END

! ALL-LABEL: func.func @_QPsubr_default_enter_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>{{.*}}
SUBROUTINE SUBR_DEFAULT_ENTER_ANY()
!$omp declare target enter(SUBR_DEFAULT_ENTER_ANY)
END

! ALL-LABEL: func.func @_QPsubr_default_any()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
SUBROUTINE SUBR_DEFAULT_ANY()
!$omp declare target
END

! ALL-LABEL: func.func @_QPsubr_default_extendedlist()
! ALL-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}
SUBROUTINE SUBR_DEFAULT_EXTENDEDLIST()
!$omp declare target(SUBR_DEFAULT_EXTENDEDLIST)
END

!! -----

! DEVICE-LABEL: func.func @_QPrecursive_declare_target
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>{{.*}}
RECURSIVE FUNCTION RECURSIVE_DECLARE_TARGET(INCREMENT) RESULT(K)
!$omp declare target to(RECURSIVE_DECLARE_TARGET) device_type(nohost)
    INTEGER :: INCREMENT, K
    IF (INCREMENT == 10) THEN
        K = INCREMENT
    ELSE
        K = RECURSIVE_DECLARE_TARGET(INCREMENT + 1)
    END IF
END FUNCTION RECURSIVE_DECLARE_TARGET

! DEVICE-LABEL: func.func @_QPrecursive_declare_target_enter
! DEVICE-SAME: {{.*}}attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>{{.*}}
RECURSIVE FUNCTION RECURSIVE_DECLARE_TARGET_ENTER(INCREMENT) RESULT(K)
!$omp declare target enter(RECURSIVE_DECLARE_TARGET_ENTER) device_type(nohost)
    INTEGER :: INCREMENT, K
    IF (INCREMENT == 10) THEN
        K = INCREMENT
    ELSE
        K = RECURSIVE_DECLARE_TARGET_ENTER(INCREMENT + 1)
    END IF
END FUNCTION RECURSIVE_DECLARE_TARGET_ENTER
