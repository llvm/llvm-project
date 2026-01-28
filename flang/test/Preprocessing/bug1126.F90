! RUN: %flang -E %s 2>&1 | FileCheck %s
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define PREFIX(x) prefix ## x
#define NAME(x) PREFIX(foo ## x)
#define AUGMENT(x) NAME(x ## suffix)

! CHECK: subroutine prefixfoosuffix()
! CHECK: print *, "prefixfoosuffix"
! CHECK: end subroutine prefixfoosuffix
subroutine AUGMENT()()
  print *, TOSTRING(AUGMENT())
end subroutine AUGMENT()

! CHECK: subroutine prefixfoobarsuffix()
! CHECK: print *, "prefixfoobarsuffix"
! CHECK: end subroutine prefixfoobarsuffix
subroutine AUGMENT(bar)()
  print *, TOSTRING(AUGMENT(bar))
end subroutine AUGMENT(bar)
