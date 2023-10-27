! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Test ignoring @PROCESS directive in free source form

@process opt(3)
@process	opt(0)
      @process strict
@processopt(3)
subroutine f()
print *, "@process"
   ! @process
end subroutine f

!CHECK: error: expected '('
@p

!CHECK: error: expected '('
@proce

!CHECK: error: expected '('
@precoss
end

