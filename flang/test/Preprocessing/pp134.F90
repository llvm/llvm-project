! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: print *, ADC, 1
! CHECK: print *, AD, 2
! CHECK: print *, DC, 3
! CHECK: print *, AD(1), 4
! CHECK: print *, AD
! CHECK: print *, AB
#define B D
implicit none
real ADC
print *, A&
  &B&
  &C, 1
print *, A&
  &B&
  &, 2
print *, &
  &B&
  &C, 3
print *, A&
  &B &
  &(1), 4
print *, A&
  &B
print *, A&
  &B ! but not this
end
