// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true bounds-in-stack-slots=true" --mlir-print-debuginfo %s | FileCheck %s
// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true" %s | FileCheck %s --check-prefix=NOSLOT

#loc1 = loc("test.f90":1:1)
#loc3 = loc("test.f90":3:14)
#loc5 = loc("test.f90":5:1)
#loc = loc("test.f90":0:0)

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func private @foo()

  func.func @test_dynamic_array(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"} loc(#loc1), %arg1: index loc(#loc1), %arg2: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_dynamic_array"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg1) origin %arg2 dummy_scope %0 arg 1 uniq_name("_QFtest_dynamic_arrayEarr") : (!fir.ref<!fir.array<?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_dynamic_array(
// CHECK:         %[[COUNT_SLOT:.*]] = fir.alloca index loc(#[[UNK:loc[0-9]+]])
// CHECK:         %[[LB_SLOT:.*]] = fir.alloca index loc(#[[UNK]])
// CHECK-NOT:     llvm.intr.dbg.value
// CHECK:         fir.store %arg2 to %[[LB_SLOT]] : !fir.ref<index> loc(#[[UNK]])
// CHECK:         fircg.ext_declare %[[LB_SLOT]] uniq_name("._QFtest_dynamic_arrayEarr2")
// CHECK:         fir.store %arg1 to %[[COUNT_SLOT]] : !fir.ref<index> loc(#[[UNK]])
// CHECK:         fircg.ext_declare %[[COUNT_SLOT]] uniq_name("._QFtest_dynamic_arrayEarr1")
// CHECK:         fircg.ext_declare %arg0(%arg1) origin %arg2
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    return

// NOSLOT-LABEL: func.func @test_dynamic_array(
// NOSLOT-NOT:    fir.alloca
// NOSLOT:        llvm.intr.dbg.value
// NOSLOT:        fir.fake_use %arg0
// NOSLOT-NEXT:   fir.fake_use %arg1
// NOSLOT-NEXT:   fir.fake_use %arg2

  func.func @test_char(%arg0: !fir.ref<!fir.char<1,?>> {fir.bindc_name = "str"} loc(#loc1), %arg1: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_char"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 typeparams %arg1 dummy_scope %0 arg 1 uniq_name("_QFtest_charEstr") : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_char(
// CHECK:         %[[LEN_SLOT:.*]] = fir.alloca index
// CHECK-NOT:     llvm.intr.dbg.value
// CHECK:         fir.store %arg1 to %[[LEN_SLOT]] : !fir.ref<index>
// CHECK:         fircg.ext_declare %[[LEN_SLOT]] uniq_name("._QFtest_charEstr1")
// CHECK:         fircg.ext_declare %arg0 typeparams %arg1
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    return

  // Both dimensions are the same value, and one slot describes them both.
  func.func @test_repeated_extent(%arg0: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "arr"} loc(#loc1), %arg1: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_repeated_extent"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg1, %arg1) dummy_scope %0 arg 1 uniq_name("_QFtest_repeated_extentEarr") : (!fir.ref<!fir.array<?x?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_repeated_extent(
// CHECK:         %[[SLOT:.*]] = fir.alloca index
// CHECK-NOT:     fir.alloca
// CHECK:         fir.store %arg1 to %[[SLOT]] : !fir.ref<index>
// CHECK:         fircg.ext_declare %[[SLOT]] uniq_name("._QFtest_repeated_extentEarr1")
// CHECK-NOT:     fircg.ext_declare %[[SLOT]]
// CHECK:         fircg.ext_declare %arg0(%arg1, %arg1)

  // Two declarations name the same value, and each gets a slot of its own, so
  // that a slot always belongs to the region of the declaration it describes.
  func.func @test_shared_extent(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a"} loc(#loc1), %arg1: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "b"} loc(#loc1), %arg2: index loc(#loc1)) attributes {fir.internal_name = "_QPtest_shared_extent"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg2) dummy_scope %0 arg 1 uniq_name("_QFtest_shared_extentEa") : (!fir.ref<!fir.array<?xi32>>, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    %2 = fircg.ext_declare %arg1(%arg2) dummy_scope %0 arg 2 uniq_name("_QFtest_shared_extentEb") : (!fir.ref<!fir.array<?xi32>>, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

// The slots go to the start of the block, so the one made last comes first.
// CHECK-LABEL: func.func @test_shared_extent(
// CHECK:         %[[B_SLOT:.*]] = fir.alloca index
// CHECK:         %[[A_SLOT:.*]] = fir.alloca index
// CHECK:         fir.store %arg2 to %[[A_SLOT]] : !fir.ref<index>
// CHECK:         fircg.ext_declare %[[A_SLOT]] uniq_name("._QFtest_shared_extentEa1")
// CHECK:         fircg.ext_declare %arg0(%arg2)
// CHECK:         fir.store %arg2 to %[[B_SLOT]] : !fir.ref<index>
// CHECK:         fircg.ext_declare %[[B_SLOT]] uniq_name("._QFtest_shared_extentEb1")
// CHECK:         fircg.ext_declare %arg1(%arg2)
// Neither declaration needs a fake use of the extent they share.
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    fir.fake_use %arg1
// CHECK-NEXT:    return

  // A declaration in a target region gets its slot in the region
  func.func @test_target_region(%arg0: !fir.ref<!fir.array<?xi32>> loc(#loc1), %arg1: !fir.ref<i32> loc(#loc1)) attributes {fir.internal_name = "_QPtest_target_region"} {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.array<?xi32>>
    %1 = omp.map.info var_ptr(%arg1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    omp.target kernel_type(generic) map_entries(%0 -> %arg2, %1 -> %arg3 : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) {
      %2 = fir.load %arg3 : !fir.ref<i32>
      %3 = fir.convert %2 : (i32) -> index
      %4 = fircg.ext_declare %arg2(%3) uniq_name("_QFtest_target_regionEarr") : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
      omp.terminator
    } loc(#loc3)
    return loc(#loc5)
  } loc(#loc1)

// CHECK-LABEL: func.func @test_target_region(
// CHECK:         omp.target
// CHECK-NEXT:      %[[TGT_SLOT:.*]] = fir.alloca index
// CHECK:           %[[EXTENT:.*]] = fir.convert %{{.*}} : (i32) -> index
// CHECK-NEXT:      fir.store %[[EXTENT]] to %[[TGT_SLOT]] : !fir.ref<index>
// CHECK-NEXT:      fircg.ext_declare %[[TGT_SLOT]] uniq_name("._QFtest_target_regionEarr1")
// CHECK-NEXT:      fircg.ext_declare %arg2(%[[EXTENT]])

// CHECK:       #[[UNK]] = loc(unknown)
} loc(#loc)
