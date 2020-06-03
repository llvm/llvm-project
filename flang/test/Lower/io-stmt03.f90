! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPcontrol0
subroutine control0(n) ! no I/O condition specifier control flow
dimension c(n), d(n,n), e(n,n), f(n)
! CHECK-NOT: fir.if
! CHECK: BeginExternalFormattedInput
! CHECK-NOT: fir.if
! CHECK: SetAdvance
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: fir.do_loop
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: fir.do_loop
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: InputReal32
! CHECK-NOT: fir.if
! CHECK: EndIoStatement
! CHECK-NOT: fir.if
read(*,'(F7.2)', advance='no') a, b, (c(j), (d(k,j), e(k,j), k=1,n), f(j), j=1,n), g
end

! CHECK-LABEL: func @_QPcontrol1
subroutine control1(n) ! I/O condition specifier control flow
! CHECK: BeginExternalFormattedInput
! CHECK: EnableHandlers
! CHECK: SetAdvance
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: fir.iterate_while
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: fir.iterate_while
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: fir.if
! CHECK: InputReal32
! CHECK: EndIoStatement
dimension c(n), d(n,n), e(n,n), f(n)
read(*,'(F7.2)', iostat=mm, advance='no') a, b, (c(j), (d(k,j), e(k,j), k=1,n), f(j), j=1,n), g
end
