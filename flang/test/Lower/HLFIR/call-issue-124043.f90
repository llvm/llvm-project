! Reproducer for https://github.com/llvm/llvm-project/issues/124043 lowering
! crash.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine repro(a)
  integer a(10)
  associate (b => a(::2)+1)
    call bar(b)
  end associate
end
! CHECK-LABEL:   func.func @_QPrepro(
! CHECK:           %[[VAL_11:.*]] = hlfir.elemental
! CHECK:           %[[VAL_16:.*]]:3 = hlfir.associate %[[VAL_11]]
! CHECK:           %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_16]]#1
! CHECK:           fir.call @_QPbar(%[[VAL_18]]#1)
