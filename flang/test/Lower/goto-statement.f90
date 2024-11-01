! RUN: bbc %s -emit-fir -hlfir=false -o - | FileCheck %s

! Test trivial goto statement
subroutine sub1()
goto 1
1 stop
end subroutine
! CHECK-LABEL: sub1
! CHECK:   cf.br ^[[BB1:.*]]
! CHECK: ^[[BB1]]:
! CHECK:   {{.*}} fir.call @_FortranAStopStatement({{.*}}, {{.*}}, {{.*}}) {{.*}}: (i32, i1, i1) -> none
! CHECK: }

! Test multiple goto statements
subroutine sub2()
goto 1
1 goto 2
2 goto 3
3 stop
end subroutine
! CHECK-LABEL: sub2
! CHECK:   cf.br ^[[BB1:.*]]
! CHECK: ^[[BB1]]:
! CHECK:   cf.br ^[[BB2:.*]]
! CHECK: ^[[BB2]]:
! CHECK:   cf.br ^[[BB3:.*]]
! CHECK: ^[[BB3]]:
! CHECK:   {{.*}} fir.call @_FortranAStopStatement({{.*}}, {{.*}}, {{.*}}) {{.*}}: (i32, i1, i1) -> none
! CHECK: }

! Test goto which branches to a previous label
subroutine sub3()
pause
1 goto 3
2 stop
3 goto 2
end subroutine
! CHECK: sub3
! CHECK:   {{.*}} fir.call @_FortranAPauseStatement() {{.*}}: () -> none
! CHECK:   cf.br ^[[BB2:.*]]
! CHECK: ^[[BB1:.*]]: //
! CHECK:   {{.*}} fir.call @_FortranAStopStatement({{.*}}, {{.*}}, {{.*}}) {{.*}}: (i32, i1, i1) -> none
! CHECK: ^[[BB2]]:
! CHECK:   cf.br ^[[BB1]]
! CHECK: }

! Test removal of blocks (pauses) which are not reachable
subroutine sub4()
pause
1 goto 2
pause
2 goto 3
pause
3 goto 1
pause
end subroutine
! CHECK-LABEL: sub4
! CHECK:   {{.*}} fir.call @_FortranAPauseStatement() {{.*}}: () -> none
! CHECK-NEXT:   cf.br ^[[BB1:.*]]
! CHECK-NEXT: ^[[BB1]]:
! CHECK-NEXT:   cf.br ^[[BB2:.*]]
! CHECK-NEXT: ^[[BB2]]:
! CHECK-NEXT:   cf.br ^[[BB3:.*]]
! CHECK-NEXT: ^[[BB3]]:
! CHECK-NEXT:   cf.br ^[[BB1]]
! CHECK-NEXT: }
