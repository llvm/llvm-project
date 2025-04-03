// RUN: fir-opt --omp-do-concurrent-conversion="map-to=device" -verify-diagnostics %s

func.func @do_concurrent_basic() attributes {fir.bindc_name = "do_concurrent_basic"} {
    %0 = fir.alloca i32 {bindc_name = "i"}
    %1:2 = hlfir.declare %0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xi32>>
    %c10 = arith.constant 10 : index
    %3 = fir.shape %c10 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %2(%3) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
    %c1_i32 = arith.constant 1 : i32
    %7 = fir.convert %c1_i32 : (i32) -> index
    %c10_i32 = arith.constant 10 : i32
    %8 = fir.convert %c10_i32 : (i32) -> index
    %c1 = arith.constant 1 : index

    // expected-error@+2 {{not yet implemented: Mapping `do concurrent` loops to device}}
    // expected-error@below {{failed to legalize operation 'fir.do_loop'}}
    fir.do_loop %arg0 = %7 to %8 step %c1 unordered {
      %13 = fir.convert %arg0 : (index) -> i32
      fir.store %13 to %1#1 : !fir.ref<i32>
      %14 = fir.load %1#0 : !fir.ref<i32>
      %15 = fir.load %1#0 : !fir.ref<i32>
      %16 = fir.convert %15 : (i32) -> i64
      %17 = hlfir.designate %4#0 (%16)  : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
      hlfir.assign %14 to %17 : i32, !fir.ref<i32>
    }

    return
  }
