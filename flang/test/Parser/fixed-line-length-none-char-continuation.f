! RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -fdebug-unparse -ffixed-line-length=132 %s | FileCheck %s --check-prefix=LEN132
! RUN: %flang_fc1 -fdebug-unparse -ffixed-line-length=none %s | FileCheck %s --check-prefix=NONE
! RUN: %flang_fc1 -fdebug-unparse -ffixed-line-length=0 %s | FileCheck %s --check-prefix=NONE
      character*4 s
      s = 'ab
     +cd'
      end
! DEFAULT: s="ab{{( {59})}}cd"
! LEN132: s="ab{{( {119})}}cd"
! NONE: s="abcd"
