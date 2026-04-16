! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcommand_only() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFcommand_onlyEcmd"}
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFcommand_onlyEcmd"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = arith.constant [[# @LINE + 8]] : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAGetCommand(%[[VAL_6]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %[[VAL_7]]) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine command_only()
  character(10) :: cmd
  call get_command(cmd)
end

! CHECK-LABEL: func.func @_QPlength_only() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "len", uniq_name = "_QFlength_onlyElen"}
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFlength_onlyElen"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = arith.constant [[# @LINE + 8]] : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_2]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAGetCommand(%[[VAL_3]], %[[VAL_6]], %[[VAL_4]], %{{.*}}, %[[VAL_7]]) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine length_only()
  integer :: len
  call get_command(length=len)
end

! CHECK-LABEL: func.func @_QPstatus_only() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFstatus_onlyEcmd"}
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFstatus_onlyEcmd"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFstatus_onlyEstat"}
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFstatus_onlyEstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_5:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_6:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_10:.*]] = arith.constant [[# @LINE + 15]] : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = fir.call @_FortranAGetCommand(%[[VAL_9]], %[[VAL_5]], %[[VAL_6]], %{{.*}}, %[[VAL_10]]) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:         fir.if %[[VAL_14]] {
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine status_only()
  character(10) :: cmd
  integer :: stat
  call get_command(cmd, status=stat)
end

! CHECK-LABEL: func.func @_QPerrmsg_only() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFerrmsg_onlyEcmd"}
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFerrmsg_onlyEcmd"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.char<1,50> {bindc_name = "err", uniq_name = "_QFerrmsg_onlyEerr"}
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} {uniq_name = "_QFerrmsg_onlyEerr"} : (!fir.ref<!fir.char<1,50>>, index) -> (!fir.ref<!fir.char<1,50>>, !fir.ref<!fir.char<1,50>>)
! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,50>>) -> !fir.box<!fir.char<1,50>>
! CHECK:         %[[VAL_6:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = arith.constant [[# @LINE + 10]] : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.char<1,50>>) -> !fir.box<none>
! CHECK:         %[[VAL_12:.*]] = fir.call @_FortranAGetCommand(%[[VAL_9]], %[[VAL_6]], %[[VAL_10]], %{{.*}}, %[[VAL_11]]) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine errmsg_only()
  character(10) :: cmd
  character(50) :: err
  call get_command(cmd, errmsg=err)
end

! CHECK-LABEL: func.func @_QPcommand_status() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFcommand_statusEcmd"}
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFcommand_statusEcmd"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFcommand_statusEstat"}
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFcommand_statusEstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_5:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_6:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_10:.*]] = arith.constant [[# @LINE + 15]] : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = fir.call @_FortranAGetCommand(%[[VAL_9]], %[[VAL_5]], %[[VAL_6]], %{{.*}}, %[[VAL_10]]) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_12]], %[[VAL_13]] : i64
! CHECK:         fir.if %[[VAL_14]] {
! CHECK:           fir.store %[[VAL_11]] to %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine command_status()
  character(10) :: cmd
  integer :: stat
  call get_command(cmd, status=stat)
end

! CHECK-LABEL: func.func @_QPall_args() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFall_argsEcmd"}
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFall_argsEcmd"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.char<1,50> {bindc_name = "err", uniq_name = "_QFall_argsEerr"}
! CHECK:         %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} {uniq_name = "_QFall_argsEerr"} : (!fir.ref<!fir.char<1,50>>, index) -> (!fir.ref<!fir.char<1,50>>, !fir.ref<!fir.char<1,50>>)
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "len", uniq_name = "_QFall_argsElen"}
! CHECK:         %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFall_argsElen"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFall_argsEstat"}
! CHECK:         %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {uniq_name = "_QFall_argsEstat"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_8:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_9:.*]] = fir.embox %[[VAL_5]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_10:.*]] = fir.embox %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,50>>) -> !fir.box<!fir.char<1,50>>
! CHECK:         %[[VAL_15:.*]] = arith.constant [[# @LINE + 18]] : i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_9]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.char<1,50>>) -> !fir.box<none>
! CHECK:         %[[VAL_16:.*]] = fir.call @_FortranAGetCommand(%[[VAL_12]], %[[VAL_13]], %[[VAL_14]], %{{.*}}, %[[VAL_15]]) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_18:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_17]], %[[VAL_18]] : i64
! CHECK:         fir.if %[[VAL_19]] {
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_7]]#0 : !fir.ref<i32>
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine all_args()
  character(10) :: cmd
  character(50) :: err
  integer :: len, stat
  call get_command(cmd, len, stat, err)
end
