! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPcommand_only() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFcommand_onlyEcmd"}
! CHECK:         %[[VAL_1:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_2:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_1]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAGetCommand(%[[VAL_6]], %[[VAL_2]], %[[VAL_3]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine command_only()
  character(10) :: cmd
  call get_command(cmd)
end

! CHECK-LABEL: func.func @_QPlength_only() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "len", uniq_name = "_QFlength_onlyElen"}
! CHECK:         %[[VAL_1:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_2:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_1]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:         %[[VAL_8:.*]] = fir.call @_FortranAGetCommand(%[[VAL_2]], %[[VAL_6]], %[[VAL_3]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine length_only()
  integer :: len
  call get_command(length=len)
end

! CHECK-LABEL: func.func @_QPstatus_only() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFstatus_onlyEcmd"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFstatus_onlyEstat"}
! CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_9:.*]] = fir.call @_FortranAGetCommand(%[[VAL_7]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:         fir.if %[[VAL_12]] {
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_1]] : !fir.ref<i32>
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
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.char<1,50> {bindc_name = "err", uniq_name = "_QFerrmsg_onlyEerr"}
! CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_3:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.char<1,50>>) -> !fir.box<!fir.char<1,50>>
! CHECK:         %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_3]] : (!fir.box<!fir.char<1,50>>) -> !fir.box<none>
! CHECK:         %[[VAL_10:.*]] = fir.call @_FortranAGetCommand(%[[VAL_7]], %[[VAL_4]], %[[VAL_8]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine errmsg_only()
  character(10) :: cmd
  character(50) :: err
  call get_command(cmd, errmsg=err)
end

! CHECK-LABEL: func.func @_QPcommand_status() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "cmd", uniq_name = "_QFcommand_statusEcmd"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFcommand_statusEstat"}
! CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_3:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_2]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_9:.*]] = fir.call @_FortranAGetCommand(%[[VAL_7]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_12:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_11]] : i64
! CHECK:         fir.if %[[VAL_12]] {
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_1]] : !fir.ref<i32>
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
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.char<1,50> {bindc_name = "err", uniq_name = "_QFall_argsEerr"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "len", uniq_name = "_QFall_argsElen"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca i32 {bindc_name = "stat", uniq_name = "_QFall_argsEstat"}
! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.char<1,50>>) -> !fir.box<!fir.char<1,50>>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.char<1,10>>) -> !fir.box<none>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_5]] : (!fir.box<i32>) -> !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.char<1,50>>) -> !fir.box<none>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranAGetCommand(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i32>) -> i64
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_15]] : i64
! CHECK:         fir.if %[[VAL_16]] {
! CHECK:           fir.store %[[VAL_13]] to %[[VAL_3]] : !fir.ref<i32>
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine all_args()
  character(10) :: cmd
  character(50) :: err
  integer :: len, stat
  call get_command(cmd, len, stat, err)
end
