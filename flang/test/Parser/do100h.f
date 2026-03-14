! RUN: %flang_fc1 -E %s | FileCheck %s
! Test that Hollerith is not mistakenly tokenized here
!CHECK: do 100 h=1,10
      do 100 h=1,10
100   continue
      end

