// Test that the default optimizer pipeline uses the allocation-placement pass
// instead of the stack-arrays and memory-allocation-opt passes. With the
// experimental -enable-allocation-placement flag, it uses the size-based
// policy. Otherwise, it emulates the legacy passes.

// RUN: tco -enable-allocation-placement %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASSES
// RUN: tco %s --mlir-pass-statistics --mlir-pass-statistics-display=pipeline -o /dev/null 2>&1 | FileCheck %s --check-prefix=PASSES
// RUN: tco -enable-allocation-placement %s | FileCheck %s --check-prefixes=NEW,CHAR
// RUN: tco %s | FileCheck %s --check-prefixes=OLD,CHAR

// REQUIRES: asserts

// PASSES-NOT: MemoryAllocationOpt
// PASSES-NOT: StackArrays
// PASSES: AllocationPlacement
// PASSES-NOT: MemoryAllocationOpt
// PASSES-NOT: StackArrays

// NEW-LABEL: define void @small_temp
// NEW: alloca [10 x i32]
// NEW-NOT: @malloc

// OLD-LABEL: define void @small_temp
// OLD: call ptr @malloc
// OLD: call void @free

func.func private @use(!fir.ref<!fir.array<10xi32>>)

func.func @small_temp() {
  %0 = fir.allocmem !fir.array<10xi32>
  %1 = fir.convert %0 : (!fir.heap<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
  fir.call @use(%1) : (!fir.ref<!fir.array<10xi32>>) -> ()
  fir.freemem %0 : !fir.heap<!fir.array<10xi32>>
  return
}

// Character expression buffers go on the stack in both cases, and their
// storage is reclaimed in each iteration of an unstructured loop.
// CHAR-LABEL: define void @character_in_cfg_loop
// CHAR: call ptr @llvm.stacksave
// CHAR-NOT: @malloc
// CHAR: alloca i8, i64
// CHAR-NOT: @malloc
// CHAR: call void @llvm.stackrestore

func.func private @extent() -> index
func.func private @use_char(!fir.ref<!fir.char<1,?>>)

func.func @character_in_cfg_loop(%again: i1) {
  cf.br ^loop
^loop:
  %n = fir.call @extent() : () -> index
  %0 = fir.allocmem !fir.char<1,?>(%n : index)
  %1 = fir.convert %0 : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  fir.call @use_char(%1) : (!fir.ref<!fir.char<1,?>>) -> ()
  fir.freemem %0 : !fir.heap<!fir.char<1,?>>
  cf.cond_br %again, ^loop, ^exit
^exit:
  return
}
