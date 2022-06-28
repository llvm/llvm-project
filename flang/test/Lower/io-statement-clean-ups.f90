! Test that any temps generated for IO options are deallocated at the right
! time (after they are used, but before exiting the block where they were
! created).
! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPtest_temp_io_options(
! CHECK-SAME:   %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}) {
subroutine test_temp_io_options(status)
  interface
    function gen_temp_character()
      character(:), allocatable :: gen_temp_character
    end function
  end interface
  integer :: status
  open (10, encoding=gen_temp_character(), file=gen_temp_character(), pad=gen_temp_character(), iostat=status)
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:  fir.call @_FortranAioBeginOpenUnit
! CHECK:  %[[VAL_15:.*]] = fir.call @_QPgen_temp_character() : () -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:  fir.save_result %[[VAL_15]] to %[[VAL_3]] : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_21:.*]] = fir.call @_FortranAioSetEncoding
! CHECK:  %[[VAL_22:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_23:.*]] = fir.box_addr %[[VAL_22]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  fir.freemem %[[VAL_23]] : !fir.heap<!fir.char<1,?>>
! CHECK:  fir.if %[[VAL_21]] {
! CHECK:    %[[VAL_27:.*]] = fir.call @_QPgen_temp_character() : () -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:    fir.save_result %[[VAL_27]] to %[[VAL_2]] : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:    %[[VAL_33:.*]] = fir.call @_FortranAioSetFile
! CHECK:    %[[VAL_34:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:    %[[VAL_35:.*]] = fir.box_addr %[[VAL_34]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:    fir.freemem %[[VAL_35]] : !fir.heap<!fir.char<1,?>>
! CHECK:    fir.if %[[VAL_33]] {
! CHECK:      %[[VAL_39:.*]] = fir.call @_QPgen_temp_character() : () -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:      fir.save_result %[[VAL_39]] to %[[VAL_1]] : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:      fir.call @_FortranAioSetPad
! CHECK:      %[[VAL_46:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:      %[[VAL_47:.*]] = fir.box_addr %[[VAL_46]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:      fir.freemem %[[VAL_47]] : !fir.heap<!fir.char<1,?>>
! CHECK:    }
! CHECK:  }
! CHECK:  fir.call @_FortranAioEndIoStatement
end subroutine
