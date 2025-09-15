!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!Regression test for bug #119005
character*2 :: ary4
!CHECK: DATA ary4/"cd"/
data ary4/"abcdef"(3:4)/
end

