! RUN: %flang -E -DMACRO= %s 2>&1 | FileCheck %s
!CHECK: integer, parameter :: p = +1
integer, parameter :: p = MACRO +1
end
