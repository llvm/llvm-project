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
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = ".result"}
! CHECK:           %[[ALLOCA_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = ".result"}
! CHECK:           %[[ALLOCA_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = ".result"}
! CHECK:           fir.call @_FortranAioBeginOpenUnit
! CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_2]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[CALL_1:.*]] = fir.call @_QPgen_temp_character() {{.*}}: () -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:           fir.save_result %[[CALL_1]] to %[[DECLARE_1]] : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           %[[EMBOXCHAR_0:.*]] = fir.emboxchar %[[BOX_ADDR_0]], %{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[BOX_ADDR_1:.*]] = fir.box_addr %[[EMBOXCHAR_0]] : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[CALL_2:.*]] = fir.call @_FortranAioSetEncoding
! CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[BOX_ADDR_1]] : (!fir.ref<!fir.char<1,?>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:           fir.freemem %[[CONVERT_3]] : !fir.heap<!fir.char<1,?>>
! CHECK:           fir.if %[[CALL_2]] {
! CHECK:             %[[DECLARE_2:.*]] = fir.declare %[[ALLOCA_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[CALL_3:.*]] = fir.call @_QPgen_temp_character() {{.*}}: () -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:             fir.save_result %[[CALL_3]] to %[[DECLARE_2]] : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DECLARE_2]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[BOX_ADDR_2:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:             %[[EMBOXCHAR_1:.*]] = fir.emboxchar %[[BOX_ADDR_2]], %{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:             %[[BOX_ADDR_3:.*]] = fir.box_addr %[[EMBOXCHAR_1]] : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
! CHECK:             %[[CALL_4:.*]] = fir.call @_FortranAioSetFile
! CHECK:             %[[CONVERT_6:.*]] = fir.convert %[[BOX_ADDR_3]] : (!fir.ref<!fir.char<1,?>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:             fir.freemem %[[CONVERT_6]] : !fir.heap<!fir.char<1,?>>
! CHECK:             fir.if %[[CALL_4]] {
! CHECK:               %[[DECLARE_3:.*]] = fir.declare %[[ALLOCA_0]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:               %[[CALL_5:.*]] = fir.call @_QPgen_temp_character() {{.*}}: () -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:               fir.save_result %[[CALL_5]] to %[[DECLARE_3]] : !fir.box<!fir.heap<!fir.char<1,?>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:               %[[LOAD_2:.*]] = fir.load %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:               %[[BOX_ADDR_4:.*]] = fir.box_addr %[[LOAD_2]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:               %[[EMBOXCHAR_2:.*]] = fir.emboxchar %[[BOX_ADDR_4]], %{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:               %[[BOX_ADDR_5:.*]] = fir.box_addr %[[EMBOXCHAR_2]] : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1,?>>
! CHECK:               fir.call @_FortranAioSetPad
! CHECK:               %[[CONVERT_9:.*]] = fir.convert %[[BOX_ADDR_5]] : (!fir.ref<!fir.char<1,?>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:               fir.freemem %[[CONVERT_9]] : !fir.heap<!fir.char<1,?>>
! CHECK:             }
! CHECK:           }
! CHECK:           fir.call @_FortranAioEndIoStatement

end subroutine
