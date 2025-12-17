! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK:error: end of file
! CHECK:in the context: END PROGRAM statement
! CHECK:unparseable.f90:9:1: in the context: main program
! CHECK:error: end of file
! CHECK:unparseable.f90:9:1: in the context: SELECT TYPE construct
module m
end
select type (barf)
