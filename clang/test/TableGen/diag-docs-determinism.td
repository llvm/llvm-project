// RUN: clang-tblgen -gen-diag-docs -I%S %s -o - 2>&1 | FileCheck %s

include "DiagnosticBase.inc"

def Pedantic : DiagGroup<"pedantic">;

// Extensions are implicitly members of -Wpedantic, and InferPedantic collects
// that membership for the docs. Collecting it into an unordered set would make
// the emitted order depend on pointer hashing, and sorting the set back by
// source location would not repair that: every record produced by a multiclass
// reports the location of the `def` inside it, so records from different
// `defm`s tie. Membership is collected into a vector instead, by marching the
// records in the order the TableGen parser holds them.
multiclass PedanticExt<string str> {
  def "" : Extension<str>;
  def _alt : Extension<str # " (alt)">;
}

defm ext_ddd : PedanticExt<"ddd is an extension">;
defm ext_bbb : PedanticExt<"bbb is an extension">;
defm ext_aaa : PedanticExt<"aaa is an extension">;
defm ext_ccc : PedanticExt<"ccc is an extension">;

// The order below is neither the `defm` order above nor a hash order: it is the
// record order, which is stable across runs.
// CHECK: aaa is an extension
// CHECK: aaa is an extension (alt)
// CHECK: bbb is an extension
// CHECK: bbb is an extension (alt)
// CHECK: ccc is an extension
// CHECK: ccc is an extension (alt)
// CHECK: ddd is an extension
// CHECK: ddd is an extension (alt)
