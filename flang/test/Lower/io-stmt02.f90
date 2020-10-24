! RUN: bbc -emit-fir -o - %s | FileCheck %s

   character*10 :: exx
   character*30 :: m
   integer*2 :: s
   exx = 'AA'
   m = 'CCCCCC'
   s = -13
   ! CHECK: call {{.*}}BeginExternalFormattedInput
   ! CHECK: call {{.*}}EnableHandlers
   ! CHECK: call {{.*}}SetAdvance
   ! CHECK: call {{.*}}InputReal
   ! CHECK: call {{.*}}GetIoMsg
   ! CHECK: call {{.*}}EndIoStatement
   ! CHECK: fir.select %{{.*}} : index [-2, ^bb4, -1, ^bb3, 0, ^bb1, unit, ^bb2]
   read(*, '(A)', ADVANCE='NO', ERR=10, END=20, EOR=30, IOSTAT=s, IOMSG=m) f
   ! CHECK-LABEL: ^bb1:
   exx = 'Zip'; goto 90
10 exx = 'Err'; goto 90
20 exx = 'End'; goto 90
30 exx = 'Eor'; goto 90
90 print*, exx, c, m, s
end

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

! CHECK-LABEL: func @_QPimpliedformat
subroutine impliedformat
  ! CHECK: BeginExternalListInput(%c-1
  ! CHECK: InputReal32
  ! CHECK: EndIoStatement(%3) : (!fir.ref<i8>) -> i32
  read*, x
  ! CHECK: BeginExternalListOutput(%c-1
  ! CHECK: OutputReal32
  ! CHECK: EndIoStatement
  print*, x
end
