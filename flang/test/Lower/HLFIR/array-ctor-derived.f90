! Test lowering of derived type array constructors to HLFIR.
! RUN: bbc -emit-hlfir --polymorphic-type -o - %s | FileCheck %s

module types
  type simple
    integer :: i
    integer :: j
  end type
end module
module derivedarrayctor
  use types
contains
  subroutine test_simple(s1, s2)
    type(simple) :: s1, s2
    call takes_simple([s1, s2])
  end subroutine
! CHECK-LABEL: func.func @_QMderivedarrayctorPtest_simple(
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>> {bindc_name = ".tmp.arrayctor"}
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}Es1"
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}Es2"
! CHECK:  %[[VAL_6:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_7:.*]] = fir.allocmem !fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, !fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>)
! CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_9]]#1(%[[VAL_8]]) : (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>>
! CHECK:  fir.store %[[VAL_10]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>>>
! CHECK:  %[[VAL_11:.*]] = arith.constant false
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[VAL_18:.*]] = fir.call @_FortranAInitArrayConstructorVector(%[[VAL_12]], %[[VAL_16]], %[[VAL_11]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, i32, !fir.ref<i8>, i32) -> none
! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.ref<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_20:.*]] = fir.call @_FortranAPushArrayConstructorSimpleScalar(%[[VAL_12]], %[[VAL_19]]) {{.*}}: (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> none
! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_22:.*]] = fir.call @_FortranAPushArrayConstructorSimpleScalar(%[[VAL_12]], %[[VAL_21]]) {{.*}}: (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> none
! CHECK:  %[[VAL_23:.*]] = arith.constant true
! CHECK:  %[[VAL_24:.*]] = hlfir.as_expr %[[VAL_9]]#0 move %[[VAL_23]] : (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, i1) -> !hlfir.expr<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>
! CHECK:  fir.call @_QMderivedarrayctorPtakes_simple
! CHECK:  hlfir.destroy %[[VAL_24]] : !hlfir.expr<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>

  subroutine test_with_polymorphic(s1, s2)
    class(simple) :: s1, s2
    call takes_simple([s1, s2])
  end subroutine
! CHECK-LABEL: func.func @_QMderivedarrayctorPtest_with_polymorphic(
! CHECK:  %[[VAL_2:.*]] = fir.alloca !fir.array<10xi64> {bindc_name = ".rt.arrayctor.vector"}
! CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>> {bindc_name = ".tmp.arrayctor"}
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}Es1"
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}Es2"
! CHECK:  %[[VAL_6:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_7:.*]] = fir.allocmem !fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>> {bindc_name = ".tmp.arrayctor", uniq_name = ""}
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_7]](%[[VAL_8]]) {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, !fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>)
! CHECK:  %[[VAL_10:.*]] = fir.embox %[[VAL_9]]#1(%[[VAL_8]]) : (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>>
! CHECK:  fir.store %[[VAL_10]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>>>
! CHECK:  %[[VAL_11:.*]] = arith.constant false
! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.array<10xi64>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[VAL_18:.*]] = fir.call @_FortranAInitArrayConstructorVector(%[[VAL_12]], %[[VAL_16]], %[[VAL_11]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.llvm_ptr<i8>, !fir.ref<!fir.box<none>>, i1, i32, !fir.ref<i8>, i32) -> none
! CHECK:  %[[VAL_19A:.*]] = fir.box_addr %[[VAL_4]]#1 : (!fir.class<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>) -> !fir.ref<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>
! CHECK:  %[[VAL_19:.*]] = fir.convert %[[VAL_19A]] : (!fir.ref<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_20:.*]] = fir.call @_FortranAPushArrayConstructorSimpleScalar(%[[VAL_12]], %[[VAL_19]]) {{.*}}: (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> none
! CHECK:  %[[VAL_21A:.*]] = fir.box_addr %[[VAL_5]]#1 : (!fir.class<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>) -> !fir.ref<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>
! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_21A]] : (!fir.ref<!fir.type<_QMtypesTsimple{i:i32,j:i32}>>) -> !fir.llvm_ptr<i8>
! CHECK:  %[[VAL_22:.*]] = fir.call @_FortranAPushArrayConstructorSimpleScalar(%[[VAL_12]], %[[VAL_21]]) {{.*}}: (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> none
! CHECK:  %[[VAL_23:.*]] = arith.constant true
! CHECK:  %[[VAL_24:.*]] = hlfir.as_expr %[[VAL_9]]#0 move %[[VAL_23]] : (!fir.heap<!fir.array<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>>, i1) -> !hlfir.expr<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>
! CHECK:  fir.call @_QMderivedarrayctorPtakes_simple
! CHECK:  hlfir.destroy %[[VAL_24]] : !hlfir.expr<2x!fir.type<_QMtypesTsimple{i:i32,j:i32}>>

  subroutine takes_simple(s)
    type(simple) :: s(:)
    print *, "got   :", s
  end subroutine
end module

  use derivedarrayctor
  type(simple) :: s1, s2
  s1%i = 1
  s1%j = 2
  s2%i = 3
  s2%j = 4

  print *, "expect: 1 2 3 4"
  call test_simple(s1, s2)
  print *, "expect: 1 2 3 4"
  call test_with_polymorphic(s1, s2)
end
