// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// The line of a DIModuleAttr comes from the location of the module's
// `fir.module_debug_imports`, which is that of the MODULE statement. Without
// one, it falls back to a guess based on the first member of the module.

module {
  // A module whose MODULE statement is on line 4, well before its first
  // member on line 20. The region is empty because the module has no USE.
  fir.module_debug_imports "no_use" {
  } loc(#loc_no_use)
  fir.global @_QMno_useEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_x)

  // A module with no members of its own, only used by `with_use` below.
  fir.module_debug_imports "other_mod" {
  } loc(#loc_other)

  // Same as `no_use`, but with a USE statement in the region.
  fir.module_debug_imports "with_use" {
    fir.use_stmt "other_mod"
  } loc(#loc_with_use)
  fir.global @_QMwith_useEy : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_y)

  // No `fir.module_debug_imports`, so the line is guessed from the member.
  fir.global @_QMlegacyEz : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  } loc(#loc_z)
}
#loc_no_use = loc("mod.f90":4:1)
#loc_x = loc("mod.f90":20:14)
#loc_other = loc("mod.f90":6:1)
#loc_with_use = loc("mod.f90":8:1)
#loc_y = loc("mod.f90":24:14)
#loc_z = loc("mod.f90":30:14)

// The MODULE statement line is used, not 19 as guessed from the member.
// CHECK-DAG: #llvm.di_module<{{.*}}name = "no_use", line = 4>

// CHECK-DAG: #llvm.di_module<{{.*}}name = "with_use", line = 8>

// Fallback when the module has no `fir.module_debug_imports`.
// CHECK-DAG: #llvm.di_module<{{.*}}name = "legacy", line = 29>
