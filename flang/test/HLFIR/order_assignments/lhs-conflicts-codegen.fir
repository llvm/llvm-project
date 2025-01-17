// Test code generation of hlfir.region_assign when the LHS computed
// address must be saved before the assignment is evaluated. Because
// the assignment would modify the LHS evaluation.
// RUN: fir-opt %s --lower-hlfir-ordered-assignments | FileCheck %s

// Test simplified IR for:
//
// x(x(1):x(2)) = l
//
// Verify that, although a conflict is detected, the LHS is not saved
// on a descriptor stack: it is already in a register that can be used
// since there is no forall.

func.func @save_box_in_ssa_register(%arg0: !fir.box<!fir.array<?xi64>>, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0:2 = hlfir.declare %arg1 {uniq_name = "l"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
  hlfir.region_assign {
    hlfir.yield %0#0 : !fir.box<!fir.array<?x!fir.logical<4>>>
  } to {
    %2 = hlfir.designate %1#0 (%c1)  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
    %3 = fir.load %2 : !fir.ref<i64>
    %4 = hlfir.designate %1#0 (%c2)  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
    %5 = fir.load %4 : !fir.ref<i64>
    %6 = arith.subi %5, %3 : i64
    %7 = fir.convert %6 : (i64) -> index
    %8 = fir.shape %7 : (index) -> !fir.shape<1>
    %9 = hlfir.designate %1#0 (%3:%5:%c1)  shape %8 : (!fir.box<!fir.array<?xi64>>, i64, i64, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi64>>
    hlfir.yield %9 : !fir.box<!fir.array<?xi64>>
  } user_defined_assign  (%arg2: !fir.ref<!fir.logical<4>>) to (%arg3: !fir.ref<i64>) {
    %2 = fir.load %arg2 : !fir.ref<!fir.logical<4>>
    fir.call @logical_to_real(%arg3, %2) : (!fir.ref<i64>, !fir.logical<4>) -> ()
  }
  return
}
// CHECK-LABEL:   func.func @save_box_in_ssa_register(
// CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xi64>>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) {
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
// CHECK:           %[[VAL_18:.*]] = hlfir.designate %[[VAL_5]]#0 (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.box<!fir.array<?xi64>>, i64, i64, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi64>>
// CHECK:           fir.do_loop %[[VAL_20:.*]] = {{.*}} {
// CHECK:             %[[VAL_21:.*]] = hlfir.designate %[[VAL_18]] (%[[VAL_20]])  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:             fir.call @logical_to_real(%[[VAL_21]], %{{.*}}) : (!fir.ref<i64>, !fir.logical<4>) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Test simplified IR for:
//
// ! x = [0,1,2,4] -> [4,2,1,1]
// forall (i=1:3) x(x(i)+1:x(i+1)) = x(4-i)
//
// Verify that the LHS are all computed an saved on a stack before
// any assignment is made.
//
func.func @save_box_in_stack(%arg0: !fir.box<!fir.array<?xi32>>) {
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c4_i64 = arith.constant 4 : i64
  %c3_i64 = arith.constant 3 : i64
  %c1_i64 = arith.constant 1 : i64
  %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  hlfir.forall lb {
    hlfir.yield %c1_i64 : i64
  } ub {
    hlfir.yield %c3_i64 : i64
  }  (%arg1: i64) {
    hlfir.region_assign {
      %1 = arith.subi %c4_i64, %arg1 : i64
      %2 = hlfir.designate %0#0 (%1)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      %3 = fir.load %2 : !fir.ref<i32>
      hlfir.yield %3 : i32
    } to {
      %1 = hlfir.designate %0#0 (%arg1)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      %2 = fir.load %1 : !fir.ref<i32>
      %3 = arith.addi %2, %c1_i32 : i32
      %4 = arith.addi %arg1, %c1_i64 : i64
      %5 = hlfir.designate %0#0 (%4)  : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
      %6 = fir.load %5 : !fir.ref<i32>
      %7 = arith.subi %6, %3 : i32
      %8 = fir.convert %7 : (i32) -> index
      %9 = fir.shape %8 : (index) -> !fir.shape<1>
      %10 = hlfir.designate %0#0 (%3:%6:%c1)  shape %9 : (!fir.box<!fir.array<?xi32>>, i32, i32, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
      hlfir.yield %10 : !fir.box<!fir.array<?xi32>>
    }
  }
  return
}
// CHECK-LABEL:   func.func @save_box_in_stack(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
// CHECK:           %[[VAL_2:.*]] = fir.alloca i64
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "x"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_30:.*]] = fir.call @_FortranACreateDescriptorStack(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK:           fir.do_loop {{.*}} {
// CHECK:             %[[VAL_48:.*]] = hlfir.designate %[[VAL_9]]#0 {{.*}}  : (!fir.box<!fir.array<?xi32>>, i32, i32, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:             fir.call @_FortranAPushDescriptor(%[[VAL_30]], %[[VAL_49]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:           }
// CHECK:           fir.store %{{.*}} to %[[VAL_2]] : !fir.ref<i64>
// CHECK:           fir.do_loop {{.*}} {
// CHECK:             %[[VAL_60:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:             %[[VAL_61:.*]] = arith.addi %[[VAL_60]], %{{.*}} : i64
// CHECK:             fir.store %[[VAL_61]] to %[[VAL_2]] : !fir.ref<i64>
// CHECK:             %[[VAL_62:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:             fir.call @_FortranADescriptorAt(%[[VAL_30]], %[[VAL_60]], %[[VAL_62]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> ()
// CHECK:             %[[VAL_64:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:             %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             hlfir.assign %{{.*}} to %[[VAL_65]] : i32, !fir.box<!fir.array<?xi32>>
// CHECK:           }
// CHECK:           fir.call @_FortranADestroyDescriptorStack(%[[VAL_30]]) : (!fir.llvm_ptr<i8>) -> ()

// Test simplified IR for:
//
// integer(8) :: x(*)
// forall (integer::i=1:10) x(x(foo(x, i):bar(x, i))) = x(11-i)
//
// The shape of the vector subscripted designator must be saved at each
// iteration.
//
func.func @test_vector_subscript_overlap(%arg0: !fir.ref<!fir.array<?xi64>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %0 = fir.undefined index
  %1 = fir.shape %0 : (index) -> !fir.shape<1>
  %2:2 = hlfir.declare %arg0(%1) {uniq_name = "x"} : (!fir.ref<!fir.array<?xi64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi64>>, !fir.ref<!fir.array<?xi64>>)
  hlfir.forall lb {
    hlfir.yield %c1 : index
  } ub {
    hlfir.yield %c10 : index
  }  (%arg1: index) {
    hlfir.region_assign {
      %3 = arith.subi %c11, %arg1 : index
      %4 = hlfir.designate %2#0 (%3)  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
      %5 = fir.load %4 : !fir.ref<i64>
      hlfir.yield %5 : i64
    } to {
      %3 = fir.call @foo(%2#1, %arg1) : (!fir.ref<!fir.array<?xi64>>, index) -> index
      %4 = fir.call @bar(%2#1, %arg1) : (!fir.ref<!fir.array<?xi64>>, index) -> index
      %5 = arith.subi %4, %3 : index
      %6 = fir.shape %5 : (index) -> !fir.shape<1>
      %7 = hlfir.designate %2#0 (%3:%4:%c1)  shape %6 : (!fir.box<!fir.array<?xi64>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi64>>
      hlfir.elemental_addr %6 : !fir.shape<1> {
      ^bb0(%arg2: index):
        %8 = hlfir.designate %7 (%arg2)  : (!fir.box<!fir.array<?xi64>>, index) -> !fir.ref<i64>
        %9 = fir.load %8 : !fir.ref<i64>
        %10 = hlfir.designate %2#0 (%9)  : (!fir.box<!fir.array<?xi64>>, i64) -> !fir.ref<i64>
        hlfir.yield %10 : !fir.ref<i64>
      }
    }
  }
  return
}
// CHECK-LABEL:   func.func @test_vector_subscript_overlap(
// CHECK-SAME:                                             %[[VAL_0:.*]]: !fir.ref<!fir.array<?xi64>>) {
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
// CHECK:           %[[VAL_2:.*]] = fir.alloca i64
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<i64>>
// CHECK:           %[[VAL_4:.*]] = fir.alloca i64
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]](%{{.*}}) {uniq_name = "x"} : (!fir.ref<!fir.array<?xi64>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi64>>, !fir.ref<!fir.array<?xi64>>)
// CHECK:           %[[VAL_30:.*]] = fir.call @_FortranACreateDescriptorStack(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK:           %[[VAL_37:.*]] = fir.call @_FortranACreateDescriptorStack(%{{.*}}, %{{.*}}) : (!fir.ref<i8>, i32) -> !fir.llvm_ptr<i8>
// CHECK:           fir.do_loop {{.*}} {
// CHECK:             %[[VAL_45:.*]] = fir.call @foo
// CHECK:             %[[VAL_46:.*]] = fir.call @bar
// CHECK:             %[[VAL_47:.*]] = arith.subi %[[VAL_46]], %[[VAL_45]] : index
// CHECK:             %[[VAL_48:.*]] = fir.shape %[[VAL_47]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_51:.*]] = fir.zero_bits !fir.ref<!fir.array<?xi32>>
// CHECK:             %[[VAL_52:.*]] = fir.embox %[[VAL_51]](%[[VAL_48]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             %[[VAL_55:.*]] = fir.convert %[[VAL_52]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// Save the vector subscripted designator shape.
// CHECK:             fir.call @_FortranAPushDescriptor({{.*}}, {{.*}}) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:             fir.do_loop {{.*}} {
// CHECK:               %[[VAL_60:.*]] = hlfir.designate %[[VAL_11]]#0 (%{{.*}})  : (!fir.box<!fir.array<?xi64>>, i64) -> !fir.ref<i64>
// CHECK:               %[[VAL_61:.*]] = fir.embox %[[VAL_60]] : (!fir.ref<i64>) -> !fir.box<i64>
// CHECK:               %[[VAL_62:.*]] = fir.convert %[[VAL_61]] : (!fir.box<i64>) -> !fir.box<none>
// Save the vector subscripted designator element address.
// CHECK:               fir.call @_FortranAPushDescriptor(%[[VAL_30]], %[[VAL_62]]) : (!fir.llvm_ptr<i8>, !fir.box<none>) -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           fir.store %{{.*}} to %[[VAL_4]] : !fir.ref<i64>
// CHECK:           fir.store %{{.*}} to %[[VAL_2]] : !fir.ref<i64>
// CHECK:           fir.do_loop {{.*}} {
// CHECK:             %[[VAL_69:.*]] = fir.load %{{.*}} : !fir.ref<i64>
// CHECK:             %[[VAL_70:.*]] = fir.load %[[VAL_2]] : !fir.ref<i64>
// CHECK:             %[[VAL_71:.*]] = arith.addi %[[VAL_70]], %{{.*}} : i64
// CHECK:             fir.store %[[VAL_71]] to %[[VAL_2]] : !fir.ref<i64>
// CHECK:             %[[VAL_72:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// Fetch the vector subscripted designator shape to create the elemental loop.
// CHECK:             fir.call @_FortranADescriptorAt(%[[VAL_37]], %[[VAL_70]], %[[VAL_72]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> ()
// CHECK:             %[[VAL_74:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
// CHECK:             %[[VAL_75:.*]] = fir.convert %[[VAL_74]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.box<!fir.array<?xi32>>
// CHECK:             %[[VAL_76:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_77:.*]]:3 = fir.box_dims %[[VAL_75]], %[[VAL_76]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
// CHECK:             %[[VAL_79:.*]] = arith.constant 1 : index
// CHECK:             fir.do_loop %[[VAL_80:.*]] = %[[VAL_79]] to %[[VAL_77]]#1 step %[[VAL_79]] {
// CHECK:               %[[VAL_81:.*]] = fir.load %[[VAL_4]] : !fir.ref<i64>
// CHECK:               %[[VAL_82:.*]] = arith.addi %[[VAL_81]], %{{.*}} : i64
// CHECK:               fir.store %[[VAL_82]] to %[[VAL_4]] : !fir.ref<i64>
// CHECK:               %[[VAL_83:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<i64>>>) -> !fir.ref<!fir.box<none>>
// Fetch the vector subscripted designator element address.
// CHECK:               fir.call @_FortranADescriptorAt(%[[VAL_30]], %[[VAL_81]], %[[VAL_83]]) : (!fir.llvm_ptr<i8>, i64, !fir.ref<!fir.box<none>>) -> ()
// CHECK:               %[[VAL_85:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<i64>>>
// CHECK:               %[[VAL_86:.*]] = fir.box_addr %[[VAL_85]] : (!fir.box<!fir.ptr<i64>>) -> !fir.ptr<i64>
// CHECK:               %[[VAL_87:.*]] = fir.convert %[[VAL_86]] : (!fir.ptr<i64>) -> !fir.ref<i64>
// CHECK:               hlfir.assign %{{.*}} to %[[VAL_87]] : i64, !fir.ref<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           fir.call @_FortranADestroyDescriptorStack(%[[VAL_30]]) : (!fir.llvm_ptr<i8>) -> ()
// CHECK:           fir.call @_FortranADestroyDescriptorStack(%[[VAL_37]]) : (!fir.llvm_ptr<i8>) -> ()

func.func private @integer_to_real(!fir.ref<i64>, !fir.logical<4>)
func.func private @foo(!fir.ref<!fir.array<?xi64>>, index) -> index
func.func private @bar(!fir.ref<!fir.array<?xi64>>, index) -> index
