! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

program p
  integer,volatile::i,arr(10)
  integer,volatile,target::tgt(10)
  integer,volatile,pointer,dimension(:)::ptr
  i = loc(i)
  ptr => tgt
  i=0
  arr=1
  call host_assoc
contains
  subroutine host_assoc
    ptr => tgt
    i=0
    arr=1
  end subroutine
end program

! CHECK-LABEL:   func.func @_QQmain() {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ADDRESS_OF_0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<volatile, internal_assoc>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:           %[[VOLATILE_CAST_1:.*]] = fir.volatile_cast %[[ALLOCA_0]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_1]] {fortran_attrs = #fir.var_attrs<volatile, internal_assoc>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VOLATILE_CAST_2:.*]] = fir.volatile_cast %[[ADDRESS_OF_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_2]] {fortran_attrs = #fir.var_attrs<pointer, volatile, internal_assoc>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[ADDRESS_OF_2:.*]] = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VOLATILE_CAST_3:.*]] = fir.volatile_cast %[[ADDRESS_OF_2]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_3]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<target, volatile, internal_assoc>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[ALLOCA_1:.*]] = fir.alloca tuple<!fir.ref<i32>>
! CHECK:           %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[ALLOCA_1]], %[[CONSTANT_1]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[VOLATILE_CAST_4:.*]] = fir.volatile_cast %[[DECLARE_1]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           fir.store %[[VOLATILE_CAST_4]] to %[[COORDINATE_OF_0]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_1]]#0 : (!fir.ref<i32, volatile>) -> i64
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[CONVERT_0]] : (i64) -> i32
! CHECK:           hlfir.assign %[[CONVERT_1]] to %[[DECLARE_1]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[DECLARE_3]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<?xi32>, volatile>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[CONVERT_2]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[EMBOX_0]] to %[[DECLARE_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DECLARE_1]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           fir.call @_QFPhost_assoc(%[[ALLOCA_1]]) fastmath<contract> : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPhost_assoc(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {{.*}} {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QFEarr) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ADDRESS_OF_0]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFEarr"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QFEptr) : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[VOLATILE_CAST_1:.*]] = fir.volatile_cast %[[ADDRESS_OF_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_1]] {fortran_attrs = #fir.var_attrs<pointer, volatile>, uniq_name = "_QFEptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>)
! CHECK:           %[[ADDRESS_OF_2:.*]] = fir.address_of(@_QFEtgt) : !fir.ref<!fir.array<10xi32>>
! CHECK:           %[[VOLATILE_CAST_2:.*]] = fir.volatile_cast %[[ADDRESS_OF_2]] : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           %[[DECLARE_2:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_2]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<target, volatile>, uniq_name = "_QFEtgt"} : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>, volatile>, !fir.ref<!fir.array<10xi32>, volatile>)
! CHECK:           %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[ARG0]], %[[CONSTANT_1]] : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[COORDINATE_OF_0]] : !fir.llvm_ptr<!fir.ref<i32>>
! CHECK:           %[[VOLATILE_CAST_3:.*]] = fir.volatile_cast %[[LOAD_0]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[DECLARE_3:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_3]] {fortran_attrs = #fir.var_attrs<volatile, host_assoc>, uniq_name = "_QFEi"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_2]]#0 : (!fir.ref<!fir.array<10xi32>, volatile>) -> !fir.ref<!fir.array<?xi32>, volatile>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[CONVERT_0]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<?xi32>, volatile>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>
! CHECK:           fir.store %[[EMBOX_0]] to %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>, volatile>, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_1]] to %[[DECLARE_3]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : i32, !fir.ref<!fir.array<10xi32>, volatile>
! CHECK:           return
! CHECK:         }
