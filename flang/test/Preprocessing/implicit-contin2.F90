! RUN: %flang -E %s | FileCheck %s
! Test implicit continuation for possible function-like macro calls
#define flm(x) x
print *, flm(1)
continue
print *, flm(2
)
end

!CHECK:      print *, 1
!CHECK:      continue
!CHECK:      print *, 2
!CHECK:      end
