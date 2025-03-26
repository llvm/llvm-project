! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! CHECK: unparseable.f90:5:1: error: parser FAIL (final position)
module m
end
select type (barf)
