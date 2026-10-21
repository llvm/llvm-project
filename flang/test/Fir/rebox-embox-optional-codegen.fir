// RUN: fir-opt --split-input-file --pass-pipeline="builtin.module(cg-rewrite)" %s | FileCheck %s

// Test that the PreCGRewrite pass (cg-rewrite) materializes a cf.cond_br
// diamond around fircg.ext_rebox / fircg.ext_embox when the original
// fir.rebox / fir.embox carries the `optional` attribute.

// CHECK-LABEL: func.func @test_rebox_optional(
// CHECK-SAME:      %[[BOX:.*]]: !fir.box<!fir.array<?xf32>>)
func.func @test_rebox_optional(%arg0: !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {
  %c0 = arith.constant 0 : index
  %0 = fir.shift %c0 : (index) -> !fir.shift<1>
  %1 = fir.rebox %arg0(%0) optional : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf32>>
  return %1 : !fir.box<!fir.array<?xf32>>
}
// CHECK:   %[[IS_PRESENT:.*]] = fir.is_present %[[BOX]] : (!fir.box<!fir.array<?xf32>>) -> i1
// CHECK:   cf.cond_br %[[IS_PRESENT]], ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK: ^[[THEN]]:
// CHECK:   %[[EXTREBOX:.*]] = fircg.ext_rebox %[[BOX]] origin %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> !fir.box<!fir.array<?xf32>>
// CHECK:   cf.br ^[[MERGE:.*]](%[[EXTREBOX]] : !fir.box<!fir.array<?xf32>>)
// CHECK: ^[[ELSE]]:
// CHECK:   %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
// CHECK:   cf.br ^[[MERGE]](%[[ABSENT]] : !fir.box<!fir.array<?xf32>>)
// CHECK: ^[[MERGE]](%[[MERGED:.*]]: !fir.box<!fir.array<?xf32>>):
// CHECK:   return %[[MERGED]] : !fir.box<!fir.array<?xf32>>

// -----

// CHECK-LABEL: func.func @test_embox_optional_dynamic(
// CHECK-SAME:      %[[REF:.*]]: !fir.ref<!fir.array<?xi32>>)
func.func @test_embox_optional_dynamic(%arg0: !fir.ref<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>> {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  %1 = fir.embox %arg0(%0) optional : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  return %1 : !fir.box<!fir.array<?xi32>>
}
// CHECK:   %[[IS_PRESENT:.*]] = fir.is_present %[[REF]] : (!fir.ref<!fir.array<?xi32>>) -> i1
// CHECK:   cf.cond_br %[[IS_PRESENT]], ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK: ^[[THEN]]:
// CHECK:   %[[EXTEMBOX:.*]] = fircg.ext_embox %[[REF]](%{{.*}}) : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.box<!fir.array<?xi32>>
// CHECK:   cf.br ^[[MERGE:.*]](%[[EXTEMBOX]] : !fir.box<!fir.array<?xi32>>)
// CHECK: ^[[ELSE]]:
// CHECK:   %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?xi32>>
// CHECK:   cf.br ^[[MERGE]](%[[ABSENT]] : !fir.box<!fir.array<?xi32>>)
// CHECK: ^[[MERGE]](%[[MERGED:.*]]: !fir.box<!fir.array<?xi32>>):
// CHECK:   return %[[MERGED]] : !fir.box<!fir.array<?xi32>>

// -----

// CHECK-LABEL: func.func @test_embox_optional_static(
// CHECK-SAME:      %[[REF:.*]]: !fir.ref<!fir.array<10xi32>>)
func.func @test_embox_optional_static(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.box<!fir.array<10xi32>> {
  %1 = fir.embox %arg0 optional : (!fir.ref<!fir.array<10xi32>>) -> !fir.box<!fir.array<10xi32>>
  return %1 : !fir.box<!fir.array<10xi32>>
}
// CHECK:   %[[IS_PRESENT:.*]] = fir.is_present %[[REF]] : (!fir.ref<!fir.array<10xi32>>) -> i1
// CHECK:   cf.cond_br %[[IS_PRESENT]], ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK: ^[[THEN]]:
// CHECK:   %[[EXTEMBOX:.*]] = fircg.ext_embox %[[REF]](%{{.*}}) : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.box<!fir.array<10xi32>>
// CHECK:   cf.br ^[[MERGE:.*]](%[[EXTEMBOX]] : !fir.box<!fir.array<10xi32>>)
// CHECK: ^[[ELSE]]:
// CHECK:   %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<10xi32>>
// CHECK:   cf.br ^[[MERGE]](%[[ABSENT]] : !fir.box<!fir.array<10xi32>>)
// CHECK: ^[[MERGE]](%[[MERGED:.*]]: !fir.box<!fir.array<10xi32>>):
// CHECK:   return %[[MERGED]] : !fir.box<!fir.array<10xi32>>

// -----

// An embox that is "kept as is" by the pass (no shape, scalar element type)
// is also wrapped in a cf.cond_br guard. The cloned embox in the "present"
// branch has its `optional` attribute dropped so that LLVM codegen sees a
// plain fir.embox.

// CHECK-LABEL: func.func @test_embox_optional_scalar_keep(
// CHECK-SAME:      %[[REF:.*]]: !fir.ref<i32>)
func.func @test_embox_optional_scalar_keep(%arg0: !fir.ref<i32>) -> !fir.box<i32> {
  %0 = fir.embox %arg0 optional : (!fir.ref<i32>) -> !fir.box<i32>
  return %0 : !fir.box<i32>
}
// CHECK:   %[[IS_PRESENT:.*]] = fir.is_present %[[REF]] : (!fir.ref<i32>) -> i1
// CHECK:   cf.cond_br %[[IS_PRESENT]], ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK: ^[[THEN]]:
// CHECK:   %[[EMBOX:.*]] = fir.embox %[[REF]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK-NOT: optional
// CHECK:   cf.br ^[[MERGE:.*]](%[[EMBOX]] : !fir.box<i32>)
// CHECK: ^[[ELSE]]:
// CHECK:   %[[ABSENT:.*]] = fir.absent !fir.box<i32>
// CHECK:   cf.br ^[[MERGE]](%[[ABSENT]] : !fir.box<i32>)
// CHECK: ^[[MERGE]](%[[MERGED:.*]]: !fir.box<i32>):
// CHECK:   return %[[MERGED]] : !fir.box<i32>
