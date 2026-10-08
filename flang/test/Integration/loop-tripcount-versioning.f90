! Check that loop trip-count versioning fires on real lowering output and is
! controlled by -f[no-]version-loops-for-trip-count. The pass matches the
! canonical form of array-section loops (arith.maxsi bound, shared slice and
! loop bound), so this test guards against lowering or canonicalization
! changes that would silently stop it from matching.
!
! RUN: %flang_fc1 -emit-llvm -O3 -fversion-loops-for-trip-count \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-tripcount-versioning \
! RUN:   %s -o /dev/null 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-llvm -O3 \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=loop-tripcount-versioning \
! RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFF --allow-empty

! The section bound is a runtime value; the declared leading extent is 4.
subroutine zero(tau, ncol)
  real(8), intent(out) :: tau(4, 26)
  integer, intent(in) :: ncol
  tau(1:ncol, :) = 0.0d0
end subroutine

! CHECK-LABEL: IR Dump After LoopTripCountVersioning
! CHECK-LABEL: func.func @_QPzero(
! CHECK:         %[[UB:.*]] = arith.maxsi %{{.*}}, %{{.*}} : index
! CHECK:         fir.do_loop
! CHECK:           %[[C4:.*]] = arith.constant 4 : index
! CHECK:           %[[CMP:.*]] = arith.cmpi eq, %[[UB]], %[[C4]] : index
! CHECK:           fir.if %[[CMP]] {
! CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[C4]] step
! CHECK:           } else {
! CHECK:             fir.do_loop %{{.*}} = %{{.*}} to %[[UB]] step
! CHECK:           }

! Without the flag the pass is not in the pipeline, so there is no IR dump.
! OFF-NOT: IR Dump After LoopTripCountVersioning
