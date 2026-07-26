// Check that fir-opt can load an out-of-tree MLIR pass plugin. This needs
// fir-opt to export its symbols (export_executable_symbols_for_plugins in
// flang/tools/fir-opt/CMakeLists.txt); without them the shared object cannot
// resolve the MLIR and FIR symbols it uses and fails to load.

// REQUIRES: plugins, examples
// XFAIL: system-aix

// RUN: fir-opt --load-pass-plugin=%llvmshlibdir/flangHLFIRPipelinePlugin%pluginext \
// RUN:   --pass-pipeline='builtin.module(print-hlfir-intrinsics{label=fir-opt})' \
// RUN:   -o /dev/null %s | FileCheck %s

// The plugin's pass must also show up in fir-opt's pass registry.
// RUN: fir-opt --load-pass-plugin=%llvmshlibdir/flangHLFIRPipelinePlugin%pluginext \
// RUN:   --help 2>&1 | FileCheck %s --check-prefix=HELP

// CHECK:      [fir-opt] begin
// CHECK:      [fir-opt] hlfir.matmul
// CHECK:      [fir-opt] hlfir.sum
// CHECK:      [fir-opt] end

// HELP: --print-hlfir-intrinsics

func.func @test_hlfir_intrinsics(%a: !fir.ref<!fir.array<3x3xf32>>,
                                 %b: !fir.ref<!fir.array<3x3xf32>>,
                                 %c: !fir.ref<!fir.array<3x3xf32>>,
                                 %s: !fir.ref<f32>) {
  %0 = hlfir.matmul %a %b : (!fir.ref<!fir.array<3x3xf32>>, !fir.ref<!fir.array<3x3xf32>>) -> !hlfir.expr<3x3xf32>
  hlfir.assign %0 to %c : !hlfir.expr<3x3xf32>, !fir.ref<!fir.array<3x3xf32>>
  hlfir.destroy %0 : !hlfir.expr<3x3xf32>
  %1 = hlfir.sum %a : (!fir.ref<!fir.array<3x3xf32>>) -> f32
  hlfir.assign %1 to %s : f32, !fir.ref<f32>
  return
}
