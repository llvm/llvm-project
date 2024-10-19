// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s


// Check if we store the correct values

func.func @wsfunc() {
  omp.parallel {
  // CHECK: fir.alloca
  // CHECK: fir.alloca
  // CHECK: fir.alloca
  // CHECK: fir.alloca
  // CHECK: fir.alloca
  // CHECK-NOT: fir.alloca
    omp.workshare {

      %t1 = "test.test1"() : () -> i32
      // CHECK: %[[T1:.*]] = "test.test1"
      // CHECK: fir.store %[[T1]]
      %t2 = "test.test2"() : () -> i32
      // CHECK: %[[T2:.*]] = "test.test2"
      // CHECK: fir.store %[[T2]]
      %t3 = "test.test3"() : () -> i32
      // CHECK: %[[T3:.*]] = "test.test3"
      // CHECK-NOT: fir.store %[[T3]]
      %t4 = "test.test4"() : () -> i32
      // CHECK: %[[T4:.*]] = "test.test4"
      // CHECK: fir.store %[[T4]]
      %t5 = "test.test5"() : () -> i32
      // CHECK: %[[T5:.*]] = "test.test5"
      // CHECK: fir.store %[[T5]]
      %t6 = "test.test6"() : () -> i32
      // CHECK: %[[T6:.*]] = "test.test6"
      // CHECK-NOT: fir.store %[[T6]]


      "test.test1"(%t1) : (i32) -> ()
      "test.test1"(%t2) : (i32) -> ()
      "test.test1"(%t3) : (i32) -> ()

      %true = arith.constant true
      fir.if %true {
        "test.test2"(%t3) : (i32) -> ()
      }

      %c1_i32 = arith.constant 1 : i32

      %t5_pure_use = arith.addi %t5, %c1_i32 : i32

      %t6_mem_effect_use = "test.test8"(%t6) : (i32) -> i32
      // CHECK: %[[T6_USE:.*]] = "test.test8"
      // CHECK: fir.store %[[T6_USE]]

      %c42 = arith.constant 42 : index
      %c1 = arith.constant 1 : index
      omp.workshare.loop_wrapper {
        omp.loop_nest (%arg1) : index = (%c1) to (%c42) inclusive step (%c1) {
          "test.test10"(%t1) : (i32) -> ()
          "test.test10"(%t5_pure_use) : (i32) -> ()
          "test.test10"(%t6_mem_effect_use) : (i32) -> ()
          omp.yield
        }
      }

      "test.test10"(%t2) : (i32) -> ()
      fir.if %true {
        "test.test10"(%t4) : (i32) -> ()
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
