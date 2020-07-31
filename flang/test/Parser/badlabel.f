! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: Label digit is not in fixed-form label field
      1 continue
! CHECK: Label digit is not in fixed-form label field
 1    2 continue
! CHECK-NOT: Label is not in fixed-form label field
      con
     3 tinue
! CHECK: Character in fixed-form label field must be a digit
end
! CHECK: 1continue
! CHECK: 12continue
! CHECK: continue
! CHECK: end
