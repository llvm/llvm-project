// Test hlfir.assign code generation to FIR of derived type requiring
// or not finalization.

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

!t_simple = !fir.type<simple{i:i32}>
fir.type_info @simple noinit nodestroy nofinal : !t_simple

func.func @test_simple(%a: !fir.ref<!t_simple>, %b: !fir.ref<!t_simple>) {
  hlfir.assign %b to %a : !fir.ref<!t_simple>, !fir.ref<!t_simple>
  return
}
// CHECK-LABEL: func.func @test_simple(
// CHECK-NOT: Destroy
// CHECK-NOT: fir.coordinate_of
// CHECK:  fir.copy %{{.*}} to %{{.*}} no_overlap : !fir.ref<!fir.type<simple{i:i32}>>, !fir.ref<!fir.type<simple{i:i32}>>


!t_with_final = !fir.type<with_final{i:i32}>
fir.type_info @with_final noinit : !t_with_final

func.func @test_with_final(%a: !fir.ref<!t_with_final>, %b: !fir.ref<!t_with_final>) {
  hlfir.assign %b to %a : !fir.ref<!t_with_final>, !fir.ref<!t_with_final>
  return
}
// CHECK-LABEL: func.func @test_with_final(
// CHECK:  fir.call @_FortranAAssign

// Derived type with two fields of different sizes: the LLVM struct will have
// tail padding.  The assignment must use fir.copy (full allocated size) rather
// than field-by-field copy so that tail-padding bytes are preserved.
!t_tail_pad = !fir.type<tail_pad,sequence{a:i32,b:i8}>
fir.type_info @tail_pad noinit nodestroy nofinal : !t_tail_pad

func.func @test_tail_pad(%a: !fir.ref<!t_tail_pad>, %b: !fir.ref<!t_tail_pad>) {
  hlfir.assign %b to %a : !fir.ref<!t_tail_pad>, !fir.ref<!t_tail_pad>
  return
}
// CHECK-LABEL: func.func @test_tail_pad(
// CHECK-NOT: fir.coordinate_of
// CHECK:  fir.copy %{{.*}} to %{{.*}} no_overlap : !fir.ref<!fir.type<tail_pad,sequence{a:i32,b:i8}>>, !fir.ref<!fir.type<tail_pad,sequence{a:i32,b:i8}>>


// Volatile derived-type assignment: the operand types must not have their
// volatility stripped by a fir.convert.  Under --strict-fir-volatile-verifier
// fir.convert rejects a conversion that drops the volatile qualifier.

// RUN: fir-opt --strict-fir-volatile-verifier --convert-hlfir-to-fir %s | FileCheck %s --check-prefix=CHECK-VOL

// Volatile LHS, non-volatile RHS.
!t_vol = !fir.type<vol_t{i:i32}>
fir.type_info @vol_t noinit nodestroy nofinal : !t_vol

func.func @test_volatile_lhs(%a: !fir.ref<!t_vol, volatile>, %b: !fir.ref<!t_vol>) {
  hlfir.assign %b to %a : !fir.ref<!t_vol>, !fir.ref<!t_vol, volatile>
  return
}
// CHECK-VOL-LABEL: func.func @test_volatile_lhs(
// CHECK-VOL-NOT:   fir.convert
// CHECK-VOL:       fir.copy %{{.*}} to %{{.*}} no_overlap : !fir.ref<!fir.type<vol_t{i:i32}>>, !fir.ref<!fir.type<vol_t{i:i32}>, volatile>

// Non-volatile LHS, volatile RHS.
func.func @test_volatile_rhs(%a: !fir.ref<!t_vol>, %b: !fir.ref<!t_vol, volatile>) {
  hlfir.assign %b to %a : !fir.ref<!t_vol, volatile>, !fir.ref<!t_vol>
  return
}
// CHECK-VOL-LABEL: func.func @test_volatile_rhs(
// CHECK-VOL-NOT:   fir.convert
// CHECK-VOL:       fir.copy %{{.*}} to %{{.*}} no_overlap : !fir.ref<!fir.type<vol_t{i:i32}>, volatile>, !fir.ref<!fir.type<vol_t{i:i32}>>

// Volatile LHS and RHS.
func.func @test_volatile_both(%a: !fir.ref<!t_vol, volatile>, %b: !fir.ref<!t_vol, volatile>) {
  hlfir.assign %b to %a : !fir.ref<!t_vol, volatile>, !fir.ref<!t_vol, volatile>
  return
}
// CHECK-VOL-LABEL: func.func @test_volatile_both(
// CHECK-VOL-NOT:   fir.convert
// CHECK-VOL:       fir.copy %{{.*}} to %{{.*}} no_overlap : !fir.ref<!fir.type<vol_t{i:i32}>, volatile>, !fir.ref<!fir.type<vol_t{i:i32}>, volatile>


// Non-SEQUENCE derived type self-assignment: lhs == rhs so alias analysis
// returns MustAlias.  noOverlap requires isTemporaryLHS OR NoAlias; neither
// holds here, so fir.copy is emitted WITHOUT no_overlap (lowers to memmove).
// disjoint is still true for non-SEQUENCE types so we take the fir.copy fast
// path rather than the _FortranAAssign runtime path.
func.func @test_simple_self_assign(%a: !fir.ref<!t_simple>) {
  hlfir.assign %a to %a : !fir.ref<!t_simple>, !fir.ref<!t_simple>
  return
}
// CHECK-LABEL: func.func @test_simple_self_assign(
// CHECK-NOT:   fir.coordinate_of
// CHECK:       fir.copy %arg0 to %arg0 : !fir.ref<!fir.type<simple{i:i32}>>, !fir.ref<!fir.type<simple{i:i32}>>
// CHECK-NOT:   no_overlap
