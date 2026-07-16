! Test forall lowering

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_forall_with_ranked_dimension() {
! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.array<10x10x!fir.type<_QFtest_forall_with_ranked_dimensionTt{arr:!fir.array<11xi32>}>>
! CHECK:         %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]]
! CHECK:         hlfir.forall lb {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         } ub {
! CHECK:           hlfir.yield %{{.*}} : i32
! CHECK:         }  (%[[I:.*]]: i32) {
! CHECK:           %[[IDX_I:.*]] = hlfir.forall_index "i" %[[I]]
! CHECK:           hlfir.region_assign {
! CHECK:             %[[RES:.*]] = fir.call @_QPf(%[[IDX_I]]) {{.*}} : (!fir.ref<i32>) -> i32
! CHECK:             hlfir.yield %[[RES]] : i32
! CHECK:           } to {
! CHECK:             %[[I_VAL:.*]] = fir.load %[[IDX_I]]
! CHECK:             %[[I_I64:.*]] = fir.convert %[[I_VAL]] : (i32) -> i64
! CHECK:             %[[A_SLICE:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[I_I64]], %{{.*}}:%{{.*}}:%{{.*}})
! CHECK:             %[[I_VAL2:.*]] = fir.load %[[IDX_I]]
! CHECK:             %[[I_OFF:.*]] = arith.addi %[[I_VAL2]], %{{.*}}
! CHECK:             %[[I_OFF_I64:.*]] = fir.convert %[[I_OFF]] : (i32) -> i64
! CHECK:             %[[A_SUB_SLICE:.*]] = hlfir.designate %[[A_SLICE]]{"arr"} {{.*}} (%[[I_OFF_I64]])
! CHECK:             hlfir.yield %[[A_SUB_SLICE]] : !fir.box<!fir.array<10xi32>>
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

subroutine test_forall_with_ranked_dimension
  interface
     pure integer function f(i)
       integer, intent(in) :: i
     end function f
  end interface
  type t
     !integer :: arr(5:15)
     integer :: arr(11)
  end type t
  type(t) :: a(10,10)

  forall (i=1:5)
     a(i,:)%arr(i+4) = f(i)
  end forall
end subroutine test_forall_with_ranked_dimension
