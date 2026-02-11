! Test that -mrecip[=<list>] works as expected.

! RUN: %flang -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-OMIT
! RUN: %flang -mrecip -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-NOARG
! RUN: %flang -mrecip=all -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-ALL
! RUN: %flang -mrecip=none -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-NONE
! RUN: %flang -mrecip=default -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-DEF
! RUN: %flang -mrecip=divd,divf,divh,vec-divd,vec-divf,vec-divh,sqrtd,sqrtf,sqrth,vec-sqrtd,vec-sqrtf,vec-sqrth -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-POS
! RUN: %flang -mrecip=!divd,!divf,!divh,!vec-divd,!vec-divf,!vec-divh,!sqrtd,!sqrtf,!sqrth,!vec-sqrtd,!vec-sqrtf,!vec-sqrth -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-NEG
! RUN: %flang -mrecip=!divd,divf,!divh,sqrtd,!sqrtf,sqrth -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG,CHECK-FLANG-MIX
! RUN: not %flang -mrecip=xxx  -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG-INV
! RUN: not %flang -mrecip=divd,divd  -### -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FLANG-DUP

! RUN: %flang_fc1 -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefix=CHECK-FC1-OMIT
! RUN: %flang_fc1 -mrecip -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1,CHECK-FC1-NOARG
! RUN: %flang_fc1 -mrecip=all -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1,CHECK-FC1-ALL
! RUN: %flang_fc1 -mrecip=none -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1,CHECK-FC1-NONE
! RUN: %flang_fc1 -mrecip=default -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1,CHECK-FC1-DEF
! RUN: %flang_fc1 -mrecip=divd,divf,divh,vec-divd,vec-divf,vec-divh,sqrtd,sqrtf,sqrth,vec-sqrtd,vec-sqrtf,vec-sqrth -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1,CHECK-FC1-POS
! RUN: %flang_fc1 -mrecip=!divd,!divf,!divh,!vec-divd,!vec-divf,!vec-divh,!sqrtd,!sqrtf,!sqrth,!vec-sqrtd,!vec-sqrtf,!vec-sqrth
! -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1,CHECK-FC1-NEG
! RUN: %flang_fc1 -mrecip=!divd,divf,!divh,sqrtd,!sqrtf,sqrth -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1,CHECK-FC1-MIX
! RUN: not %flang_fc1 -mrecip=xxx  -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1-INV
! RUN: not %flang_fc1 -mrecip=divd,divd  -emit-llvm -o - %s 2>&1| FileCheck %s --check-prefixes=CHECK-FC1-DUP

subroutine func
end subroutine func

! CHECK-FLANG: "-fc1"
! CHECK-FLANG-OMIT-NOT: "-mrecip="
! CHECK-FLANG-NOARG-SAME: "-mrecip=all"
! CHECK-FLANG-ALL-SAME: "-mrecip=all"
! CHECK-FLANG-NONE-SAME: "-mrecip=none"
! CHECK-FLANG-DEF-SAME: "-mrecip=default"
! CHECK-FLANG-POS-SAME: "-mrecip=divd,divf,divh,vec-divd,vec-divf,vec-divh,sqrtd,sqrtf,sqrth,vec-sqrtd,vec-sqrtf,vec-sqrth"
! CHECK-FLANG-NEG-SAME: "-mrecip=!divd,!divf,!divh,!vec-divd,!vec-divf,!vec-divh,!sqrtd,!sqrtf,!sqrth,!vec-sqrtd,!vec-sqrtf,!vec-sqrth"
! CHECK-FLANG-MIX-SAME: "-mrecip=!divd,divf,!divh,sqrtd,!sqrtf,sqrth"
! CHECK-FLANG-INV: error: unknown argument: 'xxx'
! CHECK-FLANG-DUP: error: invalid value 'divd' in 'mrecip='

! CHECK-FC1: define {{.+}} @func{{.*}} #[[ATTRS:[0-9]+]]
! CHECK-FC1: attributes #[[ATTRS]] =
! CHECK-FC1-OMIT-NOT: "reciprocal-estimates"
! CHECK-FC1-NOARG-SAME: "reciprocal-estimates"="all"
! CHECK-FC1-ALL-SAME: "reciprocal-estimates"="all"
! CHECK-FC1-NONE-SAME: "reciprocal-estimates"="none"
! CHECK-FC1-DEF-SAME: "reciprocal-estimates"="default"
! CHECK-FC1-POS-SAME: "reciprocal-estimates"="divd,divf,divh,vec-divd,vec-divf,vec-divh,sqrtd,sqrtf,sqrth,vec-sqrtd,vec-sqrtf,vec-sqrth"
! CHECK-FC1-NEG-SAME: "reciprocal-estimates"="!divd,!divf,!divh,!vec-divd,!vec-divf,!vec-divh,!sqrtd,!sqrtf,!sqrth,!vec-sqrtd,!vec-sqrtf,!vec-sqrth"
! CHECK-FC1-MIX-SAME: "reciprocal-estimates"="!divd,divf,!divh,sqrtd,!sqrtf,sqrth"
! CHECK-FC1-INV: error: unknown argument: 'xxx'
! CHECK-FC1-DUP: error: invalid value 'divd' in 'mrecip='
