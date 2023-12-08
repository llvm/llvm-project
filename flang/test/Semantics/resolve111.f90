!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
!Resolve names in legacy data initializers
program name
  implicit none
  integer, parameter :: bar = 1
  integer foo(bar) /bar*2/ !CHECK: foo (InDataStmt) size=4 offset=4: ObjectEntity type: INTEGER(4) shape: 1_8:1_8 init:[INTEGER(4)::2_4]
end program name
