! Ensure that unknown warning options generate a warning message. 

! RUN: %flang_fc1 -fsyntax-only -WX %s  2>&1 | FileCheck %s --check-prefix=UNKNOWN1
! RUN: %flang_fc1 -fsyntax-only -Werror2 %s  2>&1 | FileCheck %s --check-prefix=UNKNOWN2

! UNKNOWN1: unknown warning option '-WX'
! UNKNOWN2: unknown warning option '-Werror2'
