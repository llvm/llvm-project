! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_forall_with_slice(
! CHECK-SAME:       %[[VAL_0:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK:         %[[A_ALLOCA:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_sliceTt{arr:!fir.array<11xi32>}>>
! CHECK:         %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ALLOCA]]
! CHECK:         %[[I1_DECL:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK:         %[[I2_DECL:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         }  (%[[I:.*]]: i32) {
! CHECK:           %[[IDX_I:.*]] = hlfir.forall_index "i" %[[I]]
! CHECK:           hlfir.forall lb {
! CHECK:             hlfir.yield %{{.*}} : i32
! CHECK:           } ub {
! CHECK:             hlfir.yield %{{.*}} : i32
! CHECK:           }  (%[[J:.*]]: i32) {
! CHECK:             %[[IDX_J:.*]] = hlfir.forall_index "j" %[[J]]
! CHECK:             hlfir.region_assign {
! CHECK:               %[[RES:.*]] = fir.call @_QPf(%[[IDX_I]]) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK:               hlfir.yield %[[RES]] : i32
! CHECK:             } to {
! CHECK:               %[[I_VAL:.*]] = fir.load %[[IDX_I]]
! CHECK:               %[[I_I64:.*]] = fir.convert %[[I_VAL]] : (i32) -> i64
! CHECK:               %[[J_VAL:.*]] = fir.load %[[IDX_J]]
! CHECK:               %[[J_I64:.*]] = fir.convert %[[J_VAL]] : (i32) -> i64
! CHECK:               %[[A_IJ:.*]] = hlfir.designate %[[A_DECL]]#0 (%[[I_I64]], %[[J_I64]])
! CHECK:               %[[STRIDE1:.*]] = fir.load %[[I1_DECL]]#0
! CHECK:               %[[STRIDE2:.*]] = fir.load %[[I2_DECL]]#0
! CHECK:               %[[SLICE:.*]] = hlfir.designate %[[A_IJ]]{"arr"} {{.*}} (%{{.*}}:%{{.*}}:%{{.*}})
! CHECK:               hlfir.yield %[[SLICE]] : !fir.box<!fir.array<?xi32>>
! CHECK:             }
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_forall_with_slice(i1,i2)
  interface
     pure integer function f(i)
       integer i
       intent(in) i
     end function f
  end interface
  type t
     !integer :: arr(5:15)
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)

  forall (i=1:5, j=1:10)
     a(i,j)%arr(i:i1:i2) = f(i)
  end forall
end subroutine test_forall_with_slice
