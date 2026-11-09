! RUN: bbc --strict-fir-volatile-verifier %s -o - | FileCheck %s

! A pointer function result with the VOLATILE attribute must be cast to the
! non-volatile function result type when returned.

function func() result(res)
  integer, pointer :: res
  volatile :: res
  allocate(res)
  res = 7
end function

! CHECK-LABEL: func.func @_QPfunc() -> !fir.box<!fir.ptr<i32>> {
! CHECK:         %[[CAST:.*]] = fir.volatile_cast %{{.*}} : (!fir.box<!fir.ptr<i32>, volatile>) -> !fir.box<!fir.ptr<i32>>
! CHECK-NEXT:    return %[[CAST]] : !fir.box<!fir.ptr<i32>>
