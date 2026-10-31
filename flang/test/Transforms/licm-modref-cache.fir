// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s
// RUN: fir-opt -flang-licm -debug-only=flang-licm-cache --split-input-file %s \
// RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS
// REQUIRES: asserts

// Tests checking that Flang's LICM asks the alias analysis whether a loop may
// modify a location once per loop and location, and answers the other queries
// of the same run from its cache, without changing what is hoisted. The
// -debug-only=flang-licm-cache output has one line for each run of the pass
// that made queries: the number of queries answered from the cache (hits) and
// of those that asked the alias analysis (misses).

// Two loads of one location and a load of another one, none of them modified
// in the loop: the second load of the first location is a hit, and all the
// loads are hoisted.
// CHECK-LABEL: func.func @same_location_twice(
// CHECK:         %[[X:.*]] = fir.alloca i32
// CHECK:         %[[Y:.*]] = fir.alloca i32
// CHECK:         %[[Z:.*]] = fir.alloca i32
// CHECK:         %[[A:.*]] = fir.load %[[X]] : !fir.ref<i32>
// CHECK:         %[[B:.*]] = fir.load %[[X]] : !fir.ref<i32>
// CHECK:         %[[C:.*]] = fir.load %[[Y]] : !fir.ref<i32>
// CHECK:         fir.do_loop
// CHECK-NOT:       fir.load
// CHECK:           fir.store {{.*}} to %[[Z]] : !fir.ref<i32>
// STATS: flang-licm-cache: @same_location_twice: 1 hits, 2 misses
func.func @same_location_twice(%n: index) {
  %c1 = arith.constant 1 : index
  %x = fir.alloca i32
  %y = fir.alloca i32
  %z = fir.alloca i32
  fir.do_loop %i = %c1 to %n step %c1 {
    %a = fir.load %x : !fir.ref<i32>
    %b = fir.load %x : !fir.ref<i32>
    %c = fir.load %y : !fir.ref<i32>
    %s = arith.addi %a, %b : i32
    %t = arith.addi %s, %c : i32
    fir.store %t to %z : !fir.ref<i32>
  }
  return
}

// -----

// The location is modified in the loop: the answer is cached as well, so the
// second load is a hit, and neither load is hoisted.
// CHECK-LABEL: func.func @modified_location(
// CHECK:         %[[X:.*]] = fir.alloca i32
// CHECK:         fir.do_loop
// CHECK:           %[[A:.*]] = fir.load %[[X]] : !fir.ref<i32>
// CHECK:           %[[B:.*]] = fir.load %[[X]] : !fir.ref<i32>
// CHECK:           fir.store {{.*}} to %[[X]] : !fir.ref<i32>
// STATS: flang-licm-cache: @modified_location: 1 hits, 1 misses
func.func @modified_location(%n: index) {
  %c1 = arith.constant 1 : index
  %x = fir.alloca i32
  fir.do_loop %i = %c1 to %n step %c1 {
    %a = fir.load %x : !fir.ref<i32>
    %b = fir.load %x : !fir.ref<i32>
    %s = arith.addi %a, %b : i32
    fir.store %s to %x : !fir.ref<i32>
  }
  return
}

// -----

// The cache is not shared between loops: the same location asked about for
// two loops makes two queries to the alias analysis.
// CHECK-LABEL: func.func @two_loops(
// CHECK:         %[[X:.*]] = fir.alloca i32
// CHECK:         fir.load %[[X]] : !fir.ref<i32>
// CHECK:         fir.do_loop
// CHECK:         fir.load %[[X]] : !fir.ref<i32>
// CHECK:         fir.do_loop
// STATS: flang-licm-cache: @two_loops: 0 hits, 2 misses
func.func @two_loops(%n: index) {
  %c1 = arith.constant 1 : index
  %x = fir.alloca i32
  %z = fir.alloca i32
  fir.do_loop %i = %c1 to %n step %c1 {
    %a = fir.load %x : !fir.ref<i32>
    fir.store %a to %z : !fir.ref<i32>
  }
  fir.do_loop %j = %c1 to %n step %c1 {
    %b = fir.load %x : !fir.ref<i32>
    fir.store %b to %z : !fir.ref<i32>
  }
  return
}

// -----

// A run that makes no queries prints nothing.
// CHECK-LABEL: func.func @no_queries(
// STATS-NOT: @no_queries
func.func @no_queries(%n: index) {
  %c1 = arith.constant 1 : index
  %z = fir.alloca i32
  fir.do_loop %i = %c1 to %n step %c1 {
    %c = arith.constant 7 : i32
    fir.store %c to %z : !fir.ref<i32>
  }
  return
}
