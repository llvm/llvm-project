! RUN: %flang -E %s 2>&1 | FileCheck %s
#define FOO BAR
#define FO BA
#define OO AR
! CHECK: print *, BAR, 1
print *, &
  &FOO&
  &, 1
! CHECK: print *, FAR, 2
print *, &
  &F&
  &OO&
  &, 2
! CHECK: print *, BAO, 3
print *, &
  &FO&
  &O&
  &, 3
! CHECK: print *, BAR, 4
print *, &
  &F&
  &O&
  &O&
  &, 4
end
