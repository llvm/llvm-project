  ! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

  ! CHECK-LABEL: func @_QQmain() {
  ! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
  ! CHECK:         %[[A:.*]]:2 = hlfir.declare %[[VAL_0]](%{{.*}}) {uniq_name = "_QFEa"}
  ! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.array<10xf32>>
  ! CHECK:         %[[B:.*]]:2 = hlfir.declare %[[VAL_2]](%{{.*}}) {uniq_name = "_QFEb"}

  ! Statement: where (a > 4.0) b = -a
  ! CHECK:         hlfir.where {
  ! CHECK:           %[[CST_4:.*]] = arith.constant 4.000000e+00 : f32
  ! CHECK:           %[[MASK1:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
  ! CHECK:             arith.cmpf ogt, %{{.*}}, %[[CST_4]] {{.*}}: f32
  ! CHECK:           }
  ! CHECK:           hlfir.yield %[[MASK1]] : !hlfir.expr<10x!fir.logical<4>> cleanup {
  ! CHECK:             hlfir.destroy %[[MASK1]] : !hlfir.expr<10x!fir.logical<4>>
  ! CHECK:           }
  ! CHECK:         } do {
  ! CHECK:           hlfir.region_assign {
  ! CHECK:             %[[NEG:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ! CHECK:               arith.negf %{{.*}}
  ! CHECK:             }
  ! CHECK:             hlfir.yield %[[NEG]] : !hlfir.expr<10xf32>
  ! CHECK:           } to {
  ! CHECK:             hlfir.yield %[[B]]#0 : !fir.ref<!fir.array<10xf32>>
  ! CHECK:           }
  ! CHECK:         }

  ! Construct: where (a > 100.0) b = 2.0 * a
  !             elsewhere (a > 50.0) b = 3.0 + a; a = a - 1.0
  !             elsewhere a = a / 2.0
  ! CHECK:         hlfir.where {
  ! CHECK:           %[[CST_100:.*]] = arith.constant 1.000000e+02 : f32
  ! CHECK:           hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
  ! CHECK:             arith.cmpf ogt, %{{.*}}, %[[CST_100]] {{.*}}: f32
  ! CHECK:           }
  ! CHECK:         } do {
  ! CHECK:           hlfir.region_assign {
  ! CHECK:             %[[CST_2:.*]] = arith.constant 2.000000e+00 : f32
  ! CHECK:             hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ! CHECK:               arith.mulf %[[CST_2]], %{{.*}}
  ! CHECK:             }
  ! CHECK:           } to {
  ! CHECK:             hlfir.yield %[[B]]#0 : !fir.ref<!fir.array<10xf32>>
  ! CHECK:           }
  ! CHECK:           hlfir.elsewhere mask {
  ! CHECK:             %[[CST_50:.*]] = arith.constant 5.000000e+01 : f32
  ! CHECK:             hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10x!fir.logical<4>> {
  ! CHECK:               arith.cmpf ogt, %{{.*}}, %[[CST_50]] {{.*}}: f32
  ! CHECK:             }
  ! CHECK:           } do {
  ! CHECK:             hlfir.region_assign {
  ! CHECK:               %[[CST_3:.*]] = arith.constant 3.000000e+00 : f32
  ! CHECK:               hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ! CHECK:                 arith.addf %[[CST_3]], %{{.*}}
  ! CHECK:               }
  ! CHECK:             } to {
  ! CHECK:               hlfir.yield %[[B]]#0 : !fir.ref<!fir.array<10xf32>>
  ! CHECK:             }
  ! CHECK:             hlfir.region_assign {
  ! CHECK:               %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
  ! CHECK:               hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ! CHECK:                 arith.subf %{{.*}}, %[[CST_1]]
  ! CHECK:               }
  ! CHECK:             } to {
  ! CHECK:               hlfir.yield %[[A]]#0 : !fir.ref<!fir.array<10xf32>>
  ! CHECK:             }
  ! CHECK:             hlfir.elsewhere do {
  ! CHECK:               hlfir.region_assign {
  ! CHECK:                 %[[CST_2_2:.*]] = arith.constant 2.000000e+00 : f32
  ! CHECK:                 hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ! CHECK:                   arith.divf %{{.*}}, %[[CST_2_2]]
  ! CHECK:                 }
  ! CHECK:               } to {
  ! CHECK:                 hlfir.yield %[[A]]#0 : !fir.ref<!fir.array<10xf32>>
  ! CHECK:               }
  ! CHECK:             }
  ! CHECK:           }
  ! CHECK:         }
  ! CHECK:         return
  ! CHECK:       }

  real :: a(10), b(10)

  ! Statement
  where (a > 4.0) b = -a

  ! Construct
  where (a > 100.0)
     b = 2.0 * a
  elsewhere (a > 50.0)
     b = 3.0 + a
     a = a - 1.0
  elsewhere
     a = a / 2.0
  end where
end
