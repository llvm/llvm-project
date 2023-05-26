! RUN: bbc -emit-fir -o - %s | FileCheck %s
! UNSUPPORTED: system-windows

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

! CHECK-LABEL: func @_QPcontrol2
subroutine control2() ! I/O condition specifier control flow (use index result)
c = 1; d = 9
! CHECK: BeginExternalFormattedOutput
! CHECK: EnableHandlers
! CHECK: :2 = fir.iterate_while
! CHECK: = fir.if
! CHECK: OutputReal
! CHECK: = fir.if
! CHECK: OutputReal
! CHECK: fir.result
! CHECK: else
! CHECK: fir.result %false
! CHECK: fir.result
! CHECK: else
! CHECK: fir.result %false
! CHECK: = arith.addi %arg0, %c1
! CHECK: = arith.select
! CHECK: fir.result
! CHECK: fir.if %{{[0-9]*}}#1
! CHECK: OutputInteger
! CHECK: EndIoStatement
write(*,'(8F4.1,I5)',iostat=m) (c,d,j=11,14), j
end

! CHECK-LABEL: func @_QPcontrol3
subroutine control3 ! I/O condition specifier control flow
    character(10) :: internal(2) = ['aaa','bbb']
    integer stat, k(3)
    ! CHECK:   BeginInternalArrayListInput
    ! CHECK:   EnableHandlers
    ! CHECK:   InputDescriptor
    ! CHECK:   %[[V_15:[0-9]+]] = fir.call @_FortranAioEndIoStatement
    ! CHECK:   %[[V_16:[0-9]+]] = fir.convert %[[V_15]] : (i32) -> index
    ! CHECK:   fir.select %[[V_16]] : index [-2, ^bb1, -1, ^bb1, 0, ^bb1, unit, ^bb2]
    read(internal,*,err=666,iostat=stat) k ! set stat to IOSTAT_END (-1)
    ! CHECK: ^bb1:  // 3 preds: ^bb0, ^bb0, ^bb0
    ! CHECK:   StopStatementText
    ! CHECK:   fir.unreachable
    stop 'fallthrough -> ok'
    ! CHECK: ^bb2:  // pred: ^bb0
    ! CHECK:   BeginExternalListOutput
    ! CHECK:   OutputAscii
    ! CHECK:   EndIoStatement
666 print*, 'FAIL'
    end

! CHECK-LABEL: func @_QPloopnest
subroutine loopnest
   integer :: aa(3,3)
   aa = 10
   ! CHECK: BeginExternalListOutput
   ! CHECK: EnableHandlers
   ! CHECK: {{.*}}:2 = fir.iterate_while ({{.*}} = {{.*}} to {{.*}} step {{.*}}) and ({{.*}} = {{.*}}) -> (index, i1) {
   ! CHECK:   fir.if {{.*}} -> (i1) {
   ! CHECK:     {{.*}}:2 = fir.iterate_while ({{.*}} = {{.*}} to {{.*}} step {{.*}}) and ({{.*}} = {{.*}}) -> (index, i1) {
   ! CHECK:       fir.if {{.*}} -> (i1) {
   ! CHECK:         OutputInteger32
   ! CHECK:         fir.result {{.*}} : i1
   ! CHECK:       } else {
   ! CHECK:         fir.result {{.*}} : i1
   ! CHECK:       }
   ! CHECK:       fir.result {{.*}}, {{.*}} : index, i1
   ! CHECK:     }
   ! CHECK:     fir.result {{.*}}#1 : i1
   ! CHECK:   } else {
   ! CHECK:     fir.result {{.*}} : i1
   ! CHECK:   }
   ! CHECK:   fir.result {{.*}}, {{.*}} : index, i1
   ! CHECK: }
   ! CHECK: EndIoStatement
   write(*,*,err=66) ((aa(j,k)+j+k,j=1,3),k=1,3)
66 continue
end

! CHECK-LABEL: func @_QPimpliedformat
subroutine impliedformat
  ! CHECK: BeginExternalListInput(%c-1
  ! CHECK: InputReal32
  ! CHECK: EndIoStatement(%3) {{.*}}: (!fir.ref<i8>) -> i32
  read*, x
  ! CHECK: BeginExternalListOutput(%c-1
  ! CHECK: OutputReal32
  ! CHECK: EndIoStatement
  print*, x
end
