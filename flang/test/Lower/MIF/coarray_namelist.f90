! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

subroutine s()
  integer, save :: t[*]
  namelist /nl/ t
  t = 7
  write(*, nml=nl)
end subroutine

! CHECK-LABEL: func.func @_QPs() {
! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QFsEt) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] uniq_name("_QFsEt") : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>)
! CHECK: %[[C7_I32:.*]] = arith.constant 7 : i32
! CHECK: %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<i32>, corank:1>) -> !fir.heap<i32>
! CHECK: hlfir.assign %[[C7_I32]] to %[[VAL_4]] : i32, !fir.heap<i32>
! CHECK: %[[C6_I32:.*]] = arith.constant 6 : i32
! CHECK: %[[VAL_5:.*]] = fir.address_of(@_QQclX{{[0-9a-f]+}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
! CHECK: %[[C7_I32_0:.*]] = arith.constant 7 : i32
! CHECK: %[[VAL_7:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[C6_I32]], %[[VAL_6]], %[[C7_I32_0]]) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK: %[[VAL_8:.*]] = fir.address_of(@_QFsNnl) : !fir.ref<tuple<{{.*}}>>
! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<tuple<{{.*}}>>) -> !fir.ref<tuple<>>
! CHECK: %[[VAL_10:.*]] = fir.call @_FortranAioOutputNamelist(%[[VAL_7]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<tuple<>>) -> i1
! CHECK: %[[VAL_11:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_7]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK: return
! CHECK: }

! CHECK-LABEL: fir.global internal @_QFsEt : !fir.box<!fir.heap<i32>, corank:1> {
! CHECK: %[[VAL_0:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK: %[[VAL_1:.*]] = fir.embox %[[VAL_0]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>, corank:1>
! CHECK: fir.has_value %[[VAL_1]] : !fir.box<!fir.heap<i32>, corank:1>
! CHECK: }

! CHECK-LABEL: func.func @__mif_save_coarrays_allocate() {
! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! CHECK: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! CHECK: %[[VAL_2:.*]] = mif.init -> i32
! CHECK: %[[VAL_3:.*]] = fir.address_of(@_QFsEt) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
! CHECK: %[[C1_I64_0:.*]] = arith.constant 1 : i64
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! CHECK: fir.store %[[C1_I64_0]] to %[[VAL_4]] : !fir.ref<i64>
! CHECK: %[[VAL_5:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! CHECK: %[[C1_I64_1:.*]] = arith.constant 1 : i64
! CHECK: %[[VAL_6:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! CHECK: mif.alloc_coarray %[[VAL_3]] lcobounds %[[VAL_5]] ucobounds %[[VAL_6]] uniq_name("_QFsEt") : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
! CHECK: return
! CHECK: }

! CHECK-LABEL: fir.global linkonce @_QFsNnl.list <{alignment = 64 : i64}> constant : !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>> {
! CHECK: %[[VAL_0:.*]] = fir.undefined !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QQclX{{[0-9a-f]+}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
! CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
! CHECK: %[[VAL_3:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_2]], [0 : index, 0 : index] : {{.*}}
! CHECK: %[[VAL_4:.*]] = fir.address_of(@_QFsEt) : !fir.ref<!fir.box<!fir.heap<i32>, corank:1>>
! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.box<!fir.heap<i32>, corank:1>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[VAL_6:.*]] = fir.insert_value %[[VAL_3]], %[[VAL_5]], [0 : index, 1 : index] : {{.*}}
! CHECK: fir.has_value %[[VAL_6]] : !fir.array<1xtuple<!fir.ref<i8>, !fir.ref<!fir.box<none>>>>
! CHECK: }

! CHECK-LABEL: fir.global linkonce @_QFsNnl constant : tuple<{{.*}}> {
! CHECK: %[[VAL_0:.*]] = fir.undefined tuple<{{.*}}>
! CHECK: %[[VAL_1:.*]] = fir.address_of(@_QQclX{{[0-9A-F]+}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
! CHECK: %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
! CHECK: %[[VAL_3:.*]] = fir.insert_value %[[VAL_0]], %[[VAL_2]], [0 : index] : {{.*}}
! CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
! CHECK: %[[VAL_4:.*]] = fir.insert_value %[[VAL_3]], %[[C1_I64]], [1 : index] : {{.*}}
! CHECK: %[[VAL_5:.*]] = fir.address_of(@_QFsNnl.list) : {{.*}}
! CHECK: %[[VAL_6:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_5]], [2 : index] : {{.*}}
! CHECK: %[[VAL_7:.*]] = fir.address_of(@_QQdefault.nonTbpDefinedIoTable) : {{.*}}
! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : {{.*}} -> !fir.ref<none>
! CHECK: %[[VAL_9:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_8]], [3 : index] : {{.*}}
! CHECK: fir.has_value %[[VAL_9]] : tuple<{{.*}}>
! CHECK: }
