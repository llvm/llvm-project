! Test that -mrecip[=<list>] works as expected.

! RUN: %flang_fc1 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-OMIT
! RUN: %flang_fc1 -mrecip -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-NOARG
! RUN: %flang_fc1 -mrecip=all -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-ALL
! RUN: %flang_fc1 -mrecip=none -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-NONE
! RUN: %flang_fc1 -mrecip=default -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-DEF
! RUN: %flang_fc1 -mrecip=divd,divf,divh,vec-divd,vec-divf,vec-divh,sqrtd,sqrtf,sqrth,vec-sqrtd,vec-sqrtf,vec-sqrth -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-POS
! RUN: %flang_fc1 -mrecip=!divd,!divf,!divh,!vec-divd,!vec-divf,!vec-divh,!sqrtd,!sqrtf,!sqrth,!vec-sqrtd,!vec-sqrtf,!vec-sqrth
! -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-NEG
! RUN: %flang_fc1 -mrecip=!divd,divf,!divh,sqrtd,!sqrtf,sqrth -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-MIX
! RUN: not %flang_fc1 -mrecip=xxx  -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-INV
! RUN: not %flang_fc1 -mrecip=divd,divd  -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-DUP

subroutine func
end subroutine func

! CHECK-OMIT-NOT: attributes #0 = { "reciprocal-estimates"={{.*}} }
! CHECK-NOARG: attributes #0 = { "reciprocal-estimates"="all" }
! CHECK-ALL: attributes #0 = { "reciprocal-estimates"="all" }
! CHECK-NONE: attributes #0 = { "reciprocal-estimates"="none" }
! CHECK-DEF: attributes #0 = { "reciprocal-estimates"="default" }
! CHECK-POS: attributes #0 = { "reciprocal-estimates"="divd,divf,divh,vec-divd,vec-divf,vec-divh,sqrtd,sqrtf,sqrth,vec-sqrtd,vec-sqrtf,vec-sqrth" }
! CHECK-NEG: attributes #0 = { "reciprocal-estimates"="!divd,!divf,!divh,!vec-divd,!vec-divf,!vec-divh,!sqrtd,!sqrtf,!sqrth,!vec-sqrtd,!vec-sqrtf,!vec-sqrth" }
! CHECK-MIX: attributes #0 = { "reciprocal-estimates"="!divd,divf,!divh,sqrtd,!sqrtf,sqrth" }
! CHECK-INV: error: unknown argument: 'xxx'
! CHECK-DUP: error: invalid value 'divd' in 'mrecip='
