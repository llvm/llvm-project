! RUN: %flang_fc1 -fdebug-unparse -ffixed-line-length=none %s | FileCheck %s
! RUN: %flang_fc1 -fdebug-unparse -ffixed-line-length=0 %s | FileCheck %s
      integer*8 h
      data h /8Habc
     +defgh/
      end
! CHECK: DATA h/"abcdefgh"/
