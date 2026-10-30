// RUN: fir-opt %s | fir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: fir-opt --canonicalize %s | FileCheck %s --check-prefix=FOLD

func.func @asserts(%ok: i1, %n: index, %a: !fir.box<!fir.array<?xf32>>, %i: i32) -> (!fir.box<!fir.array<?xf32>>, i32) {
  fir.assert %ok, status, "allocation failed"
  %g:2 = fir.assert %ok, conformance, "extent %lld, expected %lld (100%%)" values(%n, %i : index, i32) guard(%a, %i : !fir.box<!fir.array<?xf32>>, i32)
  fir.assert %ok, argument, "argument"
  fir.assert %ok, bounds, "bounds"
  fir.assert %ok, pointer, "pointer"
  fir.assert %ok, do, "do"
  fir.assert %ok, bits, "bits"
  fir.assert %ok, divide, "divide"
  return %g#0, %g#1 : !fir.box<!fir.array<?xf32>>, i32
}
// ROUNDTRIP-LABEL: func.func @asserts
// ROUNDTRIP: fir.assert %{{.*}}, status, "allocation failed"
// ROUNDTRIP: %[[G:.*]]:2 = fir.assert %{{.*}}, conformance, "extent %lld, expected %lld (100%%)" values(%{{.*}}, %{{.*}} : index, i32) guard(%{{.*}}, %{{.*}} : !fir.box<!fir.array<?xf32>>, i32)
// ROUNDTRIP: fir.assert %{{.*}}, argument,
// ROUNDTRIP: fir.assert %{{.*}}, bounds,
// ROUNDTRIP: fir.assert %{{.*}}, pointer,
// ROUNDTRIP: fir.assert %{{.*}}, do,
// ROUNDTRIP: fir.assert %{{.*}}, bits,
// ROUNDTRIP: fir.assert %{{.*}}, divide,
// ROUNDTRIP: return %[[G]]#0, %[[G]]#1

func.func @many_values(%ok: i1, %i: i32) {
  fir.assert %ok, argument, "%lld %lld %lld %lld %lld" values(%i, %i, %i, %i, %i : i32, i32, i32, i32, i32)
  return
}
// ROUNDTRIP-LABEL: func.func @many_values
// ROUNDTRIP: fir.assert %{{.*}}, argument, "%lld %lld %lld %lld %lld" values(

func.func @true_check(%x: i32) -> i32 {
  %true = arith.constant true
  fir.assert %true, status, "no results"
  %g = fir.assert %true, argument, "guard" guard(%x : i32)
  return %g : i32
}
// FOLD-LABEL: func.func @true_check(
// FOLD-SAME: %[[X:.*]]: i32)
// FOLD-NOT: fir.assert
// FOLD: return %[[X]] : i32

func.func @false_check(%x: i32) -> i32 {
  %false = arith.constant false
  fir.assert %false, status, "no results"
  %g = fir.assert %false, argument, "guard" guard(%x : i32)
  return %g : i32
}
// FOLD-LABEL: func.func @false_check
// FOLD: fir.assert %{{.*}}, status, "no results"
// FOLD: %[[G:.*]] = fir.assert %{{.*}}, argument, "guard"
// FOLD: return %[[G]] : i32

// Unused guards must not make a potentially failing assertion trivially dead.
func.func @unused_results(%ok: i1, %x: i32) {
  %g = fir.assert %ok, bounds, "unused" guard(%x : i32)
  return
}
// FOLD-LABEL: func.func @unused_results
// FOLD: fir.assert %{{.*}}, bounds, "unused" guard

// This is not a view: folding a reboxed address would bypass the check.
func.func @keep_guard(%ok: i1, %box: !fir.box<i32>) -> !fir.box<i32> {
  %addr = fir.box_addr %box : (!fir.box<i32>) -> !fir.ref<i32>
  %guard = fir.assert %ok, pointer, "null pointer" guard(%addr : !fir.ref<i32>)
  %result = fir.embox %guard : (!fir.ref<i32>) -> !fir.box<i32>
  return %result : !fir.box<i32>
}
// FOLD-LABEL: func.func @keep_guard
// FOLD: %[[G:.*]] = fir.assert
// FOLD: fir.embox %[[G]]
