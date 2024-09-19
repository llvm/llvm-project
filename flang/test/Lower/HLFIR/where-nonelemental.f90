! Test lowering of non elemental calls and there inputs inside WHERE
! constructs. These must be lowered inside hlfir.exactly_once so that
! they are properly hoisted once the loops are materialized and
! expression evaluations are scheduled.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test_where(a, b, c)
 real, dimension(:) :: a, b, c
 interface
  function logical_func1()
    logical :: logical_func1(100)
  end function
  function logical_func2()
    logical :: logical_func2(100)
  end function
  real elemental function elem_func(x)
    real, intent(in) :: x
  end function
 end interface
 where (logical_func1())
  a = b + real_func(a+b+real_func2()) + elem_func(a)
 elsewhere(logical_func2())
  a(1:ifoo()) = c
 end where
end subroutine
! CHECK-LABEL:   func.func @_QPtest_where(
! CHECK:           hlfir.where {
! CHECK-NOT: hlfir.exactly_once
! CHECK:             %[[VAL_17:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:             %[[VAL_19:.*]] = fir.call @_QPlogical_func1() fastmath<contract> : () -> !fir.array<100x!fir.logical<4>>
! CHECK:             hlfir.yield %{{.*}} : !hlfir.expr<100x!fir.logical<4>> cleanup {
! CHECK:               llvm.intr.stackrestore %[[VAL_17]] : !llvm.ptr
! CHECK:             }
! CHECK:           } do {
! CHECK:             hlfir.region_assign {
! CHECK:               %[[VAL_24:.*]] = hlfir.exactly_once : f32 {
! CHECK:                 %[[VAL_28:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                 }
! CHECK-NOT: hlfir.exactly_once
! CHECK:                 %[[VAL_35:.*]] = fir.call @_QPreal_func2() fastmath<contract> : () -> f32
! CHECK:                 %[[VAL_36:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                 ^bb0(%[[VAL_37:.*]]: index):
! CHECK:                   %[[VAL_38:.*]] = hlfir.apply %[[VAL_28]], %[[VAL_37]] : (!hlfir.expr<?xf32>, index) -> f32
! CHECK:                   %[[VAL_39:.*]] = arith.addf %[[VAL_38]], %[[VAL_35]] fastmath<contract> : f32
! CHECK:                   hlfir.yield_element %[[VAL_39]] : f32
! CHECK:                 }
! CHECK:                 %[[VAL_41:.*]] = fir.call @_QPreal_func
! CHECK:                 hlfir.yield %[[VAL_41]] : f32 cleanup {
! CHECK:                   hlfir.destroy %[[VAL_36]] : !hlfir.expr<?xf32>
! CHECK:                   hlfir.destroy %[[VAL_28]] : !hlfir.expr<?xf32>
! CHECK:                 }
! CHECK:               }
! CHECK:               %[[VAL_45:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                 arith.addf
! CHECK-NOT: hlfir.exactly_once
! CHECK:               }
! CHECK:               %[[VAL_53:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                 fir.call @_QPelem_func
! CHECK:               }
! CHECK:               %[[VAL_57:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                 arith.addf
! CHECK:               }
! CHECK:               hlfir.yield %[[VAL_57]] : !hlfir.expr<?xf32> cleanup {
! CHECK:                 hlfir.destroy %[[VAL_57]] : !hlfir.expr<?xf32>
! CHECK:                 hlfir.destroy %[[VAL_53]] : !hlfir.expr<?xf32>
! CHECK:                 hlfir.destroy %[[VAL_45]] : !hlfir.expr<?xf32>
! CHECK:               }
! CHECK:             } to {
! CHECK:               hlfir.yield %{{.*}} : !fir.box<!fir.array<?xf32>>
! CHECK:             }
! CHECK:             hlfir.elsewhere mask {
! CHECK:               %[[VAL_62:.*]] = hlfir.exactly_once : !hlfir.expr<100x!fir.logical<4>> {
! CHECK:                 %[[VAL_72:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:                 fir.call @_QPlogical_func2() fastmath<contract> : () -> !fir.array<100x!fir.logical<4>>
! CHECK:                 hlfir.yield %{{.*}} : !hlfir.expr<100x!fir.logical<4>> cleanup {
! CHECK:                   llvm.intr.stackrestore %[[VAL_72]] : !llvm.ptr
! CHECK:                 }
! CHECK:               }
! CHECK:               hlfir.yield %[[VAL_62]] : !hlfir.expr<100x!fir.logical<4>>
! CHECK:             } do {
! CHECK:               hlfir.region_assign {
! CHECK:                 hlfir.yield %{{.*}} : !fir.box<!fir.array<?xf32>>
! CHECK:               } to {
! CHECK:                 %[[VAL_80:.*]] = hlfir.exactly_once : i32 {
! CHECK:                   %[[VAL_81:.*]] = fir.call @_QPifoo() fastmath<contract> : () -> i32
! CHECK:                   hlfir.yield %[[VAL_81]] : i32
! CHECK:                 }
! CHECK:                 hlfir.yield %{{.*}} : !fir.box<!fir.array<?xf32>>
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine test_where_in_forall(a, b, c)
 real, dimension(:, :) :: a, b, c
 interface
  pure function pure_logical_func1()
    logical :: pure_logical_func1(100)
  end function
  pure function pure_logical_func2()
    logical :: pure_logical_func2(100)
  end function
  real pure elemental function pure_elem_func(x)
    real, intent(in) :: x
  end function
  integer pure function pure_ifoo()
  end function
 end interface
 forall(i=1:10)
   where (pure_logical_func1())
    a(2*i, :) = b(i, :) + pure_real_func(a(i,:)+b(i,:)+pure_real_func2()) + pure_elem_func(a(i,:))
   elsewhere(pure_logical_func2())
    a(2*i, 1:pure_ifoo()) = c(i, :)
   end where
 end forall
end subroutine
! CHECK-LABEL:   func.func @_QPtest_where_in_forall(
! CHECK:           hlfir.forall lb {
! CHECK:             hlfir.yield %{{.*}} : i32
! CHECK:           } ub {
! CHECK:             hlfir.yield %{{.*}} : i32
! CHECK:           }  (%[[VAL_10:.*]]: i32) {
! CHECK:             %[[VAL_11:.*]] = hlfir.forall_index "i" %[[VAL_10]] : (i32) -> !fir.ref<i32>
! CHECK:             hlfir.where {
! CHECK:               %[[VAL_21:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK-NOT: hlfir.exactly_once
! CHECK:               %[[VAL_23:.*]] = fir.call @_QPpure_logical_func1() fastmath<contract> : () -> !fir.array<100x!fir.logical<4>>
! CHECK:               hlfir.yield %{{.*}} : !hlfir.expr<100x!fir.logical<4>> cleanup {
! CHECK:                 llvm.intr.stackrestore %[[VAL_21]] : !llvm.ptr
! CHECK:               }
! CHECK:             } do {
! CHECK:               hlfir.region_assign {
! CHECK:                 %[[VAL_41:.*]] = hlfir.designate
! CHECK:                 %[[VAL_42:.*]] = hlfir.exactly_once : f32 {
! CHECK:                                    hlfir.designate
! CHECK:                                    hlfir.designate
! CHECK:                   %[[VAL_71:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                     arith.addf
! CHECK:                   }
! CHECK-NOT: hlfir.exactly_once
! CHECK:                   %[[VAL_78:.*]] = fir.call @_QPpure_real_func2() fastmath<contract> : () -> f32
! CHECK:                   %[[VAL_79:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                     arith.addf
! CHECK:                   }
! CHECK:                   %[[VAL_84:.*]] = fir.call @_QPpure_real_func(
! CHECK:                   hlfir.yield %[[VAL_84]] : f32 cleanup {
! CHECK:                     hlfir.destroy %[[VAL_79]] : !hlfir.expr<?xf32>
! CHECK:                     hlfir.destroy %[[VAL_71]] : !hlfir.expr<?xf32>
! CHECK:                   }
! CHECK:                 }
! CHECK:                 %[[VAL_85:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                     arith.addf
! CHECK:                 }
! CHECK-NOT: hlfir.exactly_once
! CHECK:                 %[[VAL_104:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                 ^bb0(%[[VAL_105:.*]]: index):
! CHECK-NOT: hlfir.exactly_once
! CHECK:                   fir.call @_QPpure_elem_func
! CHECK:                 }
! CHECK:                 %[[VAL_108:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:                   arith.addf
! CHECK:                 }
! CHECK:                 hlfir.yield %[[VAL_108]] : !hlfir.expr<?xf32> cleanup {
! CHECK:                   hlfir.destroy %[[VAL_108]] : !hlfir.expr<?xf32>
! CHECK:                   hlfir.destroy %[[VAL_104]] : !hlfir.expr<?xf32>
! CHECK:                   hlfir.destroy %[[VAL_85]] : !hlfir.expr<?xf32>
! CHECK:                 }
! CHECK:               } to {
! CHECK:                 hlfir.designate
! CHECK:                 hlfir.yield %{{.*}} : !fir.box<!fir.array<?xf32>>
! CHECK:               }
! CHECK:               hlfir.elsewhere mask {
! CHECK:                 %[[VAL_129:.*]] = hlfir.exactly_once : !hlfir.expr<100x!fir.logical<4>> {
! CHECK:                   %[[VAL_139:.*]] = llvm.intr.stacksave : !llvm.ptr
! CHECK:                   %[[VAL_141:.*]] = fir.call @_QPpure_logical_func2() fastmath<contract> : () -> !fir.array<100x!fir.logical<4>>
! CHECK:                   hlfir.yield %{{.*}} : !hlfir.expr<100x!fir.logical<4>> cleanup {
! CHECK:                     llvm.intr.stackrestore %[[VAL_139]] : !llvm.ptr
! CHECK:                   }
! CHECK:                 }
! CHECK:                 hlfir.yield %[[VAL_129]] : !hlfir.expr<100x!fir.logical<4>>
! CHECK:               } do {
! CHECK:                 hlfir.region_assign {
! CHECK:                   hlfir.designate
! CHECK:                   hlfir.yield %{{.*}} : !fir.box<!fir.array<?xf32>>
! CHECK:                 } to {
! CHECK:                   %[[VAL_165:.*]] = hlfir.exactly_once : i32 {
! CHECK:                     %[[VAL_166:.*]] = fir.call @_QPpure_ifoo() fastmath<contract> : () -> i32
! CHECK:                     hlfir.yield %[[VAL_166]] : i32
! CHECK:                   }
! CHECK:                   hlfir.designate
! CHECK:                   hlfir.yield %{{.*}} : !fir.box<!fir.array<?xf32>>
! CHECK:                 }
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:           return
! CHECK:         }
