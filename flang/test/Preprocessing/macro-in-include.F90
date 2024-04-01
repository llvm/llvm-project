! RUN: %flang -I%S '-DFILE="defines.F90"' -DFOO=1 -DBAR=2 -E %s 2>&1 | FileCheck %s
#include FILE
! CHECK: integer :: a = 1
! CHECK: integer :: b = 2
#define SAME(x) x
#undef FOO
#undef BAR
#define FOO 3
#define BAR 4
#include SAME(FILE)
! CHECK: integer :: a = 3
! CHECK: integer :: b = 4
#define TOSTR(x) #x
#undef FOO
#undef BAR
#define FOO 5
#define BAR 6
#include TOSTR(defines.F90)
! CHECK: integer :: a = 5
! CHECK: integer :: b = 6
