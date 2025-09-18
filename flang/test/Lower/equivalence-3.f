! RUN: bbc -emit-fir -o - %s | FileCheck %s

  ! CHECK-LABEL: func @_QQmain
  program main
  real a1,a2
  equivalence (a1,a2)
  ! A fir.alloca should never appear in a global constant initialization.
  ! CHECK: fir.global linkonce @_QFEx1.desc constant : !fir.box<!fir.ptr<!fir.array<5xf64>>>
  ! CHECK: arith.constant 5 : index
  ! CHECK:fir.address_of(@_QFEx1) : !fir.ref<!fir.array<5xf64>>
  ! CHECK: fir.shape %c5 : (index) -> !fir.shape<1>
  ! CHECK: fir.declare %0(%1) {uniq_name = "_QFEx1"} : (!fir.ref<!fir.array<5xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<5xf64>>
  ! CHECK: fir.embox %2(%1) : (!fir.ref<!fir.array<5xf64>>, !fir.shape<1>) -> !fir.box<!fir.array<5xf64>>
  ! CHECK: fir.rebox %3 : (!fir.box<!fir.array<5xf64>>) -> !fir.box<!fir.ptr<!fir.array<5xf64>>>
  ! CHECK: fir.has_value %4 : !fir.box<!fir.ptr<!fir.array<5xf64>>>
  real*8 x1(5)
  namelist /y1/x1
  read (5,y1)
  end
