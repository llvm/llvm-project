! Test lowering of IO statement involving INTEGER(8) and INTEGER(16) external
! unit number that may turn out at runtime to be too large to fit in a default
! integer. Unit numbers must fit on default integers. This file tests that the
! related generated runtime checks and error recovery code.
! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s


! -----------------------------------------------------------------------------
!     Test that runtime checks are not emitted when not needed
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPopen_4
subroutine open_4(n4)
  integer(4) :: n4
  ! CHECK-NOT: CheckUnitNumberInRange
  ! CHECK-NOT: fir.if
  open(n4)
end subroutine

! CHECK-LABEL: func @_QPwrite_4
subroutine write_4(n4)
  integer(4) :: n4
  ! CHECK-NOT: CheckUnitNumberInRange
  ! CHECK-NOT: fir.if
  write(n4, *) "hello"
end subroutine

! CHECK-LABEL: func @_QPwrite_4_recovery
subroutine write_4_recovery(n4, ios)
  integer(4) :: n4, ios
  ! CHECK-NOT: CheckUnitNumberInRange
  write(n4, *, iostat=ios) "hello"
end subroutine

! -----------------------------------------------------------------------------
!     Test that runtime checks are emitted for integer(8/16) when there is
!     no user error recovery.
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPopen_8(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
subroutine open_8(n)
  integer(8) :: n
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.if
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_1]] : (i64) -> i32
! CHECK:  %[[VAL_13:.*]] = fir.call @_FortranAioBeginOpenUnit(%[[VAL_9]], %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:  %[[VAL_14:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_13]]) {{.*}}: (!fir.ref<i8>) -> i32
  open(n)
end subroutine

! CHECK-LABEL: func @_QPclose_8(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
subroutine close_8(n)
  integer(8) :: n
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.if
! CHECK: BeginClose
  close(n)
end subroutine

! CHECK-LABEL: func @_QPrewind_8(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
subroutine rewind_8(n)
  integer(8) :: n
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.if
! CHECK: BeginRewind
  rewind(n)
end subroutine

! CHECK-LABEL: func @_QPbackspace_8(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
subroutine backspace_8(n)
  integer(8) :: n
  backspace(n)
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.if
! CHECK: BeginBackspace
end subroutine

! CHECK-LABEL: func @_QPinquire_8(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
subroutine inquire_8(n, fm)
  integer(8) :: n
  character(*), fm
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.if
! CHECK: BeginInquire
  inquire(n, formatted=fm)
end subroutine

! CHECK-LABEL: func @_QPwrite_8(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
subroutine write_8(n)
  integer(8) :: n
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.if
! CHECK: BeginExternalListOutput
  write(n, *) "hello"
end subroutine

! CHECK-LABEL: func @_QPread_8(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
subroutine read_8(n, var)
  integer(8) :: n
  integer(4) :: var
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK-NOT: fir.if
! CHECK: BeginExternalListInput
  read(n, *) var
end subroutine

! CHECK-LABEL: func @_QPopen_16(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i128>
subroutine open_16(n)
  integer(16) :: n
! CHECK:  %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i128>
! CHECK:  %[[VAL_2:.*]] = arith.constant false
! CHECK:  %[[VAL_3:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAioCheckUnitNumberInRange128(%[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (i128, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
  open(n)
end subroutine

! -----------------------------------------------------------------------------
!     Test generation of user error recovery if-nests with INTEGER(8/16)
!     unit numbers.
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPopen_8_error_recovery_1(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
! CHECK-SAME:  , %[[VAL_1:.*]]: !fir.ref<i32>
subroutine open_8_error_recovery_1(n, ios)
  integer(8) :: n
  integer(4) :: ios
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_3:.*]] = arith.constant true
! CHECK:  %[[VAL_4:.*]] = fir.zero_bits !fir.ref<i8>
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_9:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_2]], %[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK:  %[[VAL_10:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_9]], %[[VAL_10]] : i32
! CHECK:  %[[VAL_12:.*]] = fir.if %[[VAL_11]] -> (i32) {
! CHECK:    %[[VAL_13:.*]] = fir.convert %[[VAL_2]] : (i64) -> i32
! CHECK:    %[[VAL_17:.*]] = fir.call @_FortranAioBeginOpenUnit(%[[VAL_13]], %{{.*}}, {{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:    %[[VAL_18:.*]] = arith.constant true
! CHECK:    %[[VAL_19:.*]] = arith.constant false
! CHECK:    %[[VAL_20:.*]] = arith.constant false
! CHECK:    %[[VAL_21:.*]] = arith.constant false
! CHECK:    %[[VAL_22:.*]] = arith.constant false
! CHECK:    fir.call @_FortranAioEnableHandlers(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]]) {{.*}}: (!fir.ref<i8>, i1, i1, i1, i1, i1) -> ()
! CHECK:    %[[VAL_24:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_17]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:    fir.result %[[VAL_24]] : i32
! CHECK:  } else {
! CHECK:    fir.result %[[VAL_9]] : i32
! CHECK:  }
! CHECK:  fir.store %[[VAL_25:.*]] to %[[VAL_1]] : !fir.ref<i32>
  open(n, iostat=ios)
end subroutine

! CHECK-LABEL: func @_QPopen_8_error_recovery_2(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>
! CHECK-SAME:  , %[[VAL_1:.*]]: !fir.boxchar<1>
subroutine open_8_error_recovery_2(n, msg)
  integer(8) :: n
  character(*) :: msg
! CHECK:  %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:  %[[VAL_4:.*]] = arith.constant true
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
! CHECK:  %[[VAL_10:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(%[[VAL_3]], %[[VAL_4]], %[[VAL_5]], %[[VAL_6]], %{{.*}}, %{{.*}}) {{.*}}: (i64, i1, !fir.ref<i8>, i64, !fir.ref<i8>, i32) -> i32
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:  %[[VAL_13:.*]] = fir.if %[[VAL_12]] -> (i32) {
! CHECK:    %[[VAL_14:.*]] = fir.convert %[[VAL_3]] : (i64) -> i32
! CHECK:    %[[VAL_18:.*]] = fir.call @_FortranAioBeginOpenUnit(%[[VAL_14]], %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:    %[[VAL_19:.*]] = arith.constant false
! CHECK:    %[[VAL_20:.*]] = arith.constant true
! CHECK:    %[[VAL_21:.*]] = arith.constant false
! CHECK:    %[[VAL_22:.*]] = arith.constant false
! CHECK:    %[[VAL_23:.*]] = arith.constant true
! CHECK:    fir.call @_FortranAioEnableHandlers(%[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]]) {{.*}}: (!fir.ref<i8>, i1, i1, i1, i1, i1) -> ()
! CHECK:    %[[VAL_25:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:    %[[VAL_26:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
! CHECK:    fir.call @_FortranAioGetIoMsg(%[[VAL_18]], %[[VAL_25]], %[[VAL_26]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64) -> ()
! CHECK:    %[[VAL_28:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_18]]) {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:    fir.result %[[VAL_28]] : i32
! CHECK:  } else {
! CHECK:    fir.result %[[VAL_10]] : i32
! CHECK:  }
! CHECK:  %[[VAL_29:.*]] = fir.convert %[[VAL_30:.*]] : (i32) -> index
! CHECK:  fir.select %[[VAL_29]] : index [0, ^bb1, unit, ^bb2]
! CHECK:       ^bb1:
! CHECK:  br ^bb3
! CHECK:       ^bb2:
! CHECK:  fir.call @_QPi_failed() {{.*}}: () -> ()
! CHECK:  br ^bb3
! CHECK:       ^bb3:
! CHECK:  return
  open(n, err=30, iomsg=msg)
  return
30 call i_failed()
end subroutine

! Torture test for temp clean-ups when user recovery is enabled.
! Checks that temps are cleaned-up in the right nests.
subroutine temp_cleanup(n, msg, ios)
  interface
    function make_temp0()
      integer, allocatable :: make_temp0
    end function
    function make_temp1()
      integer, allocatable :: make_temp1
    end function
    function make_temp2()
      integer, allocatable :: make_temp2
    end function
    function make_temp3()
      integer, allocatable :: make_temp3
    end function
    function make_temp4()
      integer, allocatable :: make_temp4
    end function
    function make_temp5()
      integer, allocatable :: make_temp5
    end function
  end interface
  integer(8) :: n(2)
  character(80) :: msg
  Integer ios(2)
  write(n(make_temp0()), iostat=ios(make_temp1()), iomsg=msg(make_temp2():make_temp3())) make_temp4(), make_temp5()
! CHECK-LABEL: func @_QPtemp_cleanup(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<2xi64>> {fir.bindc_name = "n"},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "msg"},
! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "ios"}) {
! CHECK:  %[[VAL_10:.*]] = fir.call @_QPmake_temp2() {{.*}}: () -> !fir.box<!fir.heap<i32>>
! CHECK:  fir.save_result %[[VAL_10]] to %[[VAL_8:.*]] : !fir.box<!fir.heap<i32>>, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_15:.*]] = fir.call @_QPmake_temp3() {{.*}}: () -> !fir.box<!fir.heap<i32>>
! CHECK:  fir.save_result %[[VAL_15]] to %[[VAL_7:.*]] : !fir.box<!fir.heap<i32>>, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  fir.load %[[VAL_7]]
! CHECK:  %[[VAL_32:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_33:.*]] = fir.box_addr %[[VAL_32]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  fir.freemem %[[VAL_33]] : !fir.heap<i32>
! CHECK:  %[[VAL_37:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_38:.*]] = fir.box_addr %[[VAL_37]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  fir.freemem %[[VAL_38]] : !fir.heap<i32>
! CHECK:  %[[VAL_42:.*]] = fir.call @_QPmake_temp0() {{.*}}: () -> !fir.box<!fir.heap<i32>>
! CHECK:  fir.save_result %[[VAL_42]] to %[[VAL_6:.*]] : !fir.box<!fir.heap<i32>>, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_57:.*]] = fir.call @_FortranAioCheckUnitNumberInRange64(
! CHECK:  %[[VAL_58:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_59:.*]] = arith.cmpi eq, %[[VAL_57]], %[[VAL_58]] : i32
! CHECK:  %[[VAL_60:.*]] = fir.if %[[VAL_59]] -> (i32) {
! CHECK:    fir.call @_FortranAioBeginUnformattedOutput(
! CHECK:    fir.call @_FortranAioEnableHandlers(
! CHECK:    %[[VAL_72:.*]] = fir.call @_QPmake_temp4() {{.*}}: () -> !fir.box<!fir.heap<i32>>
! CHECK:    fir.save_result %[[VAL_72]] to %[[VAL_5:.*]] : !fir.box<!fir.heap<i32>>, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:    %[[VAL_77:.*]] = fir.call @_FortranAioOutputDescriptor(
! CHECK:    %[[VAL_77_1:.*]] = fir.load %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:    %[[VAL_77_2:.*]] = fir.box_addr %[[VAL_77_1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:    fir.freemem %[[VAL_77_2]] : !fir.heap<i32>
! CHECK:    fir.if %[[VAL_77]] {
! CHECK:      %[[VAL_78:.*]] = fir.call @_QPmake_temp5() {{.*}}: () -> !fir.box<!fir.heap<i32>>
! CHECK:      fir.save_result %[[VAL_78]] to %[[VAL_4:.*]] : !fir.box<!fir.heap<i32>>, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:      fir.call @_FortranAioOutputDescriptor(
! CHECK:      %[[VAL_84:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:      %[[VAL_85:.*]] = fir.box_addr %[[VAL_84]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:      fir.freemem %[[VAL_85]] : !fir.heap<i32>
! CHECK:    }
! CHECK-NOT: fir.call @_QPmake_temp3
! CHECK:    fir.call @_FortranAioGetIoMsg(
! CHECK:    %[[VAL_97:.*]] = fir.call @_FortranAioEndIoStatement(
! CHECK:    fir.result %[[VAL_97]] : i32
! CHECK:  } else {
! CHECK:    fir.result %[[VAL_57]] : i32
! CHECK:  }
! CHECK:  %[[VAL_98:.*]] = fir.call @_QPmake_temp1() {{.*}}: () -> !fir.box<!fir.heap<i32>>
! CHECK:  fir.save_result %[[VAL_98]] to %[[VAL_3:.*]] : !fir.box<!fir.heap<i32>>, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  fir.load %[[VAL_3]]
! CHECK:  %[[VAL_107:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_108:.*]] = fir.box_addr %[[VAL_107]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  fir.freemem %[[VAL_108]] : !fir.heap<i32>
! CHECK:  %[[VAL_112:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[VAL_113:.*]] = fir.box_addr %[[VAL_112]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  fir.freemem %[[VAL_113]] : !fir.heap<i32>
end
