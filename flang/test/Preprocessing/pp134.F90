! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: print *, ADC
#define B D
implicit none
real ADC
print *, A&
  &B&
  &C
end
