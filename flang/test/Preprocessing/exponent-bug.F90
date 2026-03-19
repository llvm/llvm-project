! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: 3.14159e00
#define e00 e666
print *, 3.14159e00
end
