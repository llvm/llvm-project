! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -fsyntax-only %t/apd1.f
! RUN: %flang_fc1 -E -P %t/apd1.f | FileCheck %t/apd1.f --implicit-check-not=@ --implicit-check-not=process --implicit-check-not=opt
! RUN: not %flang_fc1 -fsyntax-only %t/apd2.f 2>&1 | FileCheck %t/apd2.f --implicit-check-not=error:
! RUN: not %flang_fc1 -fsyntax-only %t/apd3.f 2>&1 | FileCheck %t/apd3.f

! Test ignoring @PROCESS directive in fixed source form
! apd1.f: the directive must start at column 1 or after column 6
! apd2.f: error scenarios: start at column 6, incomplete directives
! apd3.f: incomplete directive at column 1, scan aborts

!--- apd1.f
@process opt(3)
@process	opt(0)
@process
@processopt(3)
      @process opt(3)
      @process  opt(0)
      @process
      @processopt(3)
                     @process opt(3)
! tab
	@process opt(3)
	@process
      subroutine f()
c@process
*@process
!@process
      end
! CHECK: subroutine f()
! CHECK-NEXT: end

!--- apd2.f
! CHECK: error: Could not parse {{.*}}apd2.f
! CHECK: apd2.f:{{[0-9]+}}:6: warning: {{.*}}
! CHECK: apd2.f:{{[0-9]+}}:19: error: {{.*}}
     @process opt(3)
! CHECK: apd2.f:{{[0-9]+}}:8: error: {{.*}}
      @
! CHECK: apd2.f:{{[0-9]+}}:15: error: {{.*}}
! CHECK: apd2.f:{{[0-9]+}}:21: error: {{.*}}
        @proce
      subroutine g()
      end

!--- apd3.f
!CHECK: Character in fixed-form label field must be a digit
@

!CHECK: Character in fixed-form label field must be a digit
@proce 

!CHECK: Character in fixed-form label field must be a digit
@precoss 

      end
