! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.


program p
  use iso_fortran_env, only: event_type
  class(event_type), allocatable :: e[:]
  allocate(event_type :: e[*])
  event post(e[1])
  event wait(e)
end program

! COARRAY-LABEL: func.func @_QQmain() 
! COARRAY: %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
! COARRAY: %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
! COARRAY: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
! COARRAY: %[[VAL_3:.*]] = fir.address_of(@_QFEe) : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFEe") fortran_attrs<allocatable> : {{.*}}
! COARRAY: %[[VAL_5:.*]] = fir.absent !fir.box<none>
! COARRAY: %[[VAL_6:.*]] = fir.type_desc !fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>
! COARRAY: %[[C0_I32:.*]] = arith.constant 0 : i32
! COARRAY: %[[C0_I32_0:.*]] = arith.constant 0 : i32
! COARRAY: %[[VAL_7:.*]] = fir.convert %[[VAL_4]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
! COARRAY: %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : {{.*}} -> !fir.ref<none>
! COARRAY: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[VAL_7]], %[[VAL_8]], %[[C0_I32]], %[[C0_I32_0]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! COARRAY: %[[C1_I64:.*]] = arith.constant 1 : i64
! COARRAY: %[[C0:.*]] = arith.constant 0 : index
! COARRAY: %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
! COARRAY: fir.store %[[C1_I64]] to %[[VAL_9]] : !fir.ref<i64>
! COARRAY: %[[VAL_10:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
! COARRAY: %[[VAL_11:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
! COARRAY: mif.alloc_coarray %[[VAL_4]]#0 lcobounds %[[VAL_10]] ucobounds %[[VAL_11]] errmsg %[[VAL_5]] uniq_name("_QFEe") : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
! COARRAY: %[[VAL_12:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: %[[C1_I64_1:.*]] = arith.constant 1 : i64
! COARRAY: mif.event_post %[[VAL_12]][%[[C1_I64_1]]] : (!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>, i64) -> ()
! COARRAY: %[[VAL_13:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
! COARRAY: mif.event_wait %[[VAL_13]] : (!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>) -> ()
! COARRAY: return
! COARRAY: }
