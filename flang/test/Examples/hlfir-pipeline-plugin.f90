! Check that a plugin loaded with `flang -fc1 -load` can insert MLIR passes into
! the HLFIR-to-FIR pass pipeline through the HLFIR extension points.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! At -O0 no HLFIR simplification runs, so the intrinsics reach both extension
! points. -emit-fir exercises CodeGenAction::lowerHLFIRToFIR.
! RUN: %flang_fc1 -load %llvmshlibdir/flangHLFIRPipelinePlugin%pluginext \
! RUN:   -emit-fir -o /dev/null %s 2>&1 | FileCheck %s

! The same callbacks must fire on the -emit-llvm path, which builds its own
! config in CodeGenAction::generateLLVMIR.
! RUN: %flang_fc1 -load %llvmshlibdir/flangHLFIRPipelinePlugin%pluginext \
! RUN:   -emit-llvm -o /dev/null %s 2>&1 | FileCheck %s

! At -O2 SimplifyHLFIRIntrinsics expands the intrinsics before the Last
! extension point, which is why both extension points exist.
! RUN: %flang_fc1 -load %llvmshlibdir/flangHLFIRPipelinePlugin%pluginext \
! RUN:   -O2 -emit-fir -o /dev/null %s 2>&1 | FileCheck %s --check-prefix=O2

! Without the plugin nothing is printed.
! RUN: %flang_fc1 -emit-fir -o /dev/null %s 2>&1 \
! RUN:   | FileCheck %s --check-prefix=NOPLUGIN --allow-empty

! CHECK:      [hlfir-early] begin
! CHECK:      [hlfir-early] hlfir.matmul
! CHECK:      [hlfir-early] hlfir.sum
! CHECK:      [hlfir-early] end
! CHECK:      [hlfir-last] begin
! CHECK:      [hlfir-last] hlfir.matmul
! CHECK:      [hlfir-last] hlfir.sum
! CHECK:      [hlfir-last] end

! O2:      [hlfir-early] begin
! O2:      [hlfir-early] hlfir.matmul
! O2:      [hlfir-early] hlfir.sum
! O2:      [hlfir-early] end
! O2:      [hlfir-last] begin
! O2-NOT:  hlfir.matmul
! O2-NOT:  hlfir.sum
! O2:      [hlfir-last] end

! NOPLUGIN-NOT: hlfir-early
! NOPLUGIN-NOT: hlfir-last

subroutine test_hlfir_intrinsics(a, b, c, s)
  real :: a(3,3), b(3,3), c(3,3), s
  c = matmul(a, b)
  s = sum(a)
end subroutine
