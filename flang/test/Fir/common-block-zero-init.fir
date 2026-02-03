  // RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s
  
  fir.global @cdb_com_ {acc.declare = #acc.declare<dataClause =  acc_create>, alignment = 4 : i64} : tuple<!fir.array<100xi32>> {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.zero_bits tuple<!fir.array<100xi32>>
    %1 = fir.undefined !fir.array<100xi32>
    %2 = fir.insert_on_range %1, %c0_i32 from (0) to (99) : (!fir.array<100xi32>, i32) -> !fir.array<100xi32>
    %3 = fir.insert_value %0, %2, [0 : index] : (tuple<!fir.array<100xi32>>, !fir.array<100xi32>) -> tuple<!fir.array<100xi32>>
    fir.has_value %3 : tuple<!fir.array<100xi32>>
  }

// CHECK-LABEL: @cdb_com_ = common global { [100 x i32] } zeroinitializer
