!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

character(kind=4), parameter :: c(2) = [character(kind=4) :: &
4_'🍌', 4_'水' ]
print *, '🍌'
print *, 4_'🍌'
print *, '水'
print *, 4_'水'
end

!CHECK: CHARACTER(KIND=4_4), PARAMETER :: c(2_4) = [CHARACTER(KIND=4,LEN=1)::4_"\360\237\215\214",4_"\346\260\264"]
!CHECK: PRINT *, "\360\237\215\214"
!CHECK: PRINT *, 4_"\360\237\215\214"
!CHECK: PRINT *, "\346\260\264"
!CHECK: PRINT *, 4_"\346\260\264"
