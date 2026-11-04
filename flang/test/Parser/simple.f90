! Check that SIMPLE is recognized in the parse tree
! RUN: %flang_fc1 -fdebug-dump-parse-tree %s | FileCheck %s

simple function foo()
  return
end function

! CHECK: Simple
