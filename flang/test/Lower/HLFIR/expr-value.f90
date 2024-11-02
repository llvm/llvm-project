! Test lowering of of expressions as values
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPfoo()
subroutine foo()
  print *, 42
  ! CHECK: %[[c42:.*]] = arith.constant 42 : i32
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[c42]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
end subroutine

! CHECK-LABEL: func.func @_QPfoo_designator(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i32>
subroutine foo_designator(n)
  !CHECK:  %[[n:.*]]:2 = hlfir.declare %[[arg0]] {uniq_name = "_QFfoo_designatorEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  print *, n
  ! CHECK: %[[nval:.*]] = fir.load %[[n]]#1 : !fir.ref<i32>
  ! CHECK: fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[nval]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
end subroutine
