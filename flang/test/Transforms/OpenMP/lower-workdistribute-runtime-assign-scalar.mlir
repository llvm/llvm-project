// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// Test lowering of workdistribute for a scalar assignment within a target teams workdistribute region.
// The test checks that the scalar assignment is correctly lowered to wsloop and loop_nest operations.

// Example Fortran code:
// !$omp target teams workdistribute
// y = 3.0_real32
// !$omp end target teams workdistribute


// CHECK-LABEL:   func.func @x(
// CHECK:             omp.target {{.*}} {
// CHECK:               omp.teams {
// CHECK:                 omp.parallel {
// CHECK:                   omp.distribute {
// CHECK:                     omp.wsloop {
// CHECK:                       omp.loop_nest (%[[VAL_73:.*]]) : index = (%[[VAL_66:.*]]) to (%[[VAL_72:.*]]) inclusive step (%[[VAL_67:.*]]) {
// CHECK:                         %[[VAL_74:.*]] = arith.constant 0 : index
// CHECK:                         %[[VAL_75:.*]]:3 = fir.box_dims %[[VAL_64:.*]], %[[VAL_74]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:                         %[[VAL_76:.*]] = arith.constant 1 : index
// CHECK:                         %[[VAL_77:.*]]:3 = fir.box_dims %[[VAL_64]], %[[VAL_76]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:                         %[[VAL_78:.*]] = arith.constant 1 : index
// CHECK:                         %[[VAL_79:.*]] = arith.remsi %[[VAL_73]], %[[VAL_77]]#1 : index
// CHECK:                         %[[VAL_80:.*]] = arith.addi %[[VAL_79]], %[[VAL_78]] : index
// CHECK:                         %[[VAL_81:.*]] = arith.divsi %[[VAL_73]], %[[VAL_77]]#1 : index
// CHECK:                         %[[VAL_82:.*]] = arith.remsi %[[VAL_81]], %[[VAL_75]]#1 : index
// CHECK:                         %[[VAL_83:.*]] = arith.addi %[[VAL_82]], %[[VAL_78]] : index
// CHECK:                         %[[VAL_84:.*]] = fir.array_coor %[[VAL_64]] %[[VAL_83]], %[[VAL_80]] : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
// CHECK:                         fir.store %[[VAL_65:.*]] to %[[VAL_84]] : !fir.ref<f32>
// CHECK:                         omp.yield
// CHECK:                       }
// CHECK:                     } {omp.composite}
// CHECK:                   } {omp.composite}
// CHECK:                   omp.terminator
// CHECK:                 } {omp.composite}
// CHECK:                 omp.terminator
// CHECK:               }
// CHECK:               omp.terminator
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime}

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
func.func @x(%arr : !fir.ref<!fir.array<?x?xf32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c78 = arith.constant 78 : index
    %cst = arith.constant 3.000000e+00 : f32
    %0 = fir.alloca i32
    %1 = fir.alloca i32
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index
    %194 = arith.subi %c10, %c1 : index
    %195 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%194 : index) extent(%c10 : index) stride(%c1 : index) start_idx(%c1 : index)
    %196 = arith.subi %c20, %c1 : index
    %197 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%196 : index) extent(%c20 : index) stride(%c1 : index) start_idx(%c1 : index)
    %198 = omp.map.info var_ptr(%arr : !fir.ref<!fir.array<?x?xf32>>, f32) map_clauses(implicit, tofrom) capture(ByRef) bounds(%195, %197) -> !fir.ref<!fir.array<?x?xf32>> {name = "y"}
    %199 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    %200 = omp.map.info var_ptr(%0 : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = ""}
    omp.target map_entries(%198 -> %arg5, %199 -> %arg6, %200 -> %arg7 : !fir.ref<!fir.array<?x?xf32>>, !fir.ref<i32>, !fir.ref<i32>) {
      %c0_0 = arith.constant 0 : index
      %201 = fir.load %arg7 : !fir.ref<i32>
      %202 = fir.load %arg6 : !fir.ref<i32>
      %203 = fir.convert %202 : (i32) -> i64
      %204 = fir.convert %201 : (i32) -> i64
      %205 = fir.convert %204 : (i64) -> index
      %206 = arith.cmpi sgt, %205, %c0_0 : index
      %207 = fir.convert %203 : (i64) -> index
      %208 = arith.cmpi sgt, %207, %c0_0 : index
      %209 = arith.select %208, %207, %c0_0 : index
      %210 = arith.select %206, %205, %c0_0 : index
      %211 = fir.shape %210, %209 : (index, index) -> !fir.shape<2>
      %212 = fir.declare %arg5(%211) {uniq_name = "_QFFaxpy_array_workdistributeEy"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.ref<!fir.array<?x?xf32>>
      %213 = fir.embox %212(%211) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
      omp.teams {
        %214 = fir.alloca !fir.box<!fir.array<?x?xf32>> {pinned}
        omp.workdistribute {
          %215 = fir.alloca f32
          %216 = fir.embox %215 : (!fir.ref<f32>) -> !fir.box<f32>
          %217 = fir.shape %210, %209 : (index, index) -> !fir.shape<2>
          %218 = fir.embox %212(%217) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
          fir.store %218 to %214 : !fir.ref<!fir.box<!fir.array<?x?xf32>>>
          %219 = fir.address_of(@_QQclXf9c642d28e5bba1f07fa9a090b72f4fc) : !fir.ref<!fir.char<1,78>>
          %c39_i32 = arith.constant 39 : i32
          %220 = fir.convert %214 : (!fir.ref<!fir.box<!fir.array<?x?xf32>>>) -> !fir.ref<!fir.box<none>>
          %221 = fir.convert %216 : (!fir.box<f32>) -> !fir.box<none>
          %222 = fir.convert %219 : (!fir.ref<!fir.char<1,78>>) -> !fir.ref<i8>
          fir.call @_FortranAAssign(%220, %221, %222, %c39_i32) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    return
}

func.func private @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) attributes {fir.runtime}

fir.global linkonce @_QQclXf9c642d28e5bba1f07fa9a090b72f4fc constant : !fir.char<1,78> {
  %0 = fir.string_lit "File: /work/github/skc7/llvm-project/build_fomp_reldebinfo/saxpy_tests/\00"(78) : !fir.char<1,78>
  fir.has_value %0 : !fir.char<1,78>
}
}
