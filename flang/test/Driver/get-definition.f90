! Verify that the driver correctly rejects invalid values for -fget-definition

! RUN: not %flang_fc1 -fget-definition 36 1 2 %s 2>&1 | FileCheck --check-prefix=OK %s
! RUN: not %flang_fc1 -fget-definition a 1 1 %s 2>&1 | FileCheck --check-prefix=ERROR-a %s
! RUN: not %flang_fc1 -fget-definition 1 b 1 %s 2>&1 | FileCheck --check-prefix=ERROR-b %s
! RUN: not %flang_fc1 -fget-definition 1 1 c %s 2>&1 | FileCheck --check-prefix=ERROR-c %s
! RUN: not %flang_fc1 -fget-definition a b 1 %s 2>&1 | FileCheck --check-prefix=ERROR-ab %s
! RUN: not %flang_fc1 -fget-definition a b c %s 2>&1 | FileCheck --check-prefix=ERROR-abc %s
! RUN: not %flang_fc1 -fget-definition 1 b c %s 2>&1 | FileCheck --check-prefix=ERROR-bc %s
! RUN: not %flang_fc1 -fget-definition a 1 c %s 2>&1 | FileCheck --check-prefix=ERROR-ac %s

! OK: String range: >m<
! OK-NOT: error

! ERROR-a: error: invalid value 'a' in 'fget-definition'
! ERROR-a-NOT: String range: >m<

! ERROR-b: error: invalid value 'b' in 'fget-definition'
! ERROR-b-NOT: String range: >m<

! ERROR-c: error: invalid value 'c' in 'fget-definition'
! ERROR-c-NOT: String range: >m<

! ERROR-ab: error: invalid value 'a' in 'fget-definition'
! ERROR-ab-NOT: String range: >m<

! ERROR-ac: error: invalid value 'a' in 'fget-definition'
! ERROR-ac-NOT: String range: >m<

! ERROR-bc: error: invalid value 'b' in 'fget-definition'
! ERROR-bc-NOT: String range: >m<

! ERROR-abc: error: invalid value 'a' in 'fget-definition'
! ERROR-abc-NOT: String range: >m<

module m
end module
