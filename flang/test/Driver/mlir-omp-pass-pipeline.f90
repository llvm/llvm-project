! Test the MLIR pass pipeline for OpenMP

! RUN: %flang_fc1 -S -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefix=NO_OMP %s
! RUN: %flang_fc1 -S -fopenmp -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefix=FULL %s
! RUN: %flang_fc1 -S -fopenmp -fopenmp-simd -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefix=FULL %s
! RUN: %flang_fc1 -S -fopenmp-simd -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefix=SIMD %s
! RUN: %flang_fc1 -S -fopenmp -fopenmp-is-target-device -mmlir --mlir-pass-statistics -mmlir --mlir-pass-statistics-display=pipeline -o /dev/null %s 2>&1 | FileCheck --check-prefix=DEVICE %s

! REQUIRES: asserts

end program

! NO_OMP-NOT: MapsForPrivatizedSymbolsPass
! NO_OMP-NOT: AutomapToTargetDataPass
! NO_OMP-NOT: MapInfoFinalizationPass
! NO_OMP-NOT: GenericLoopConversionPass
! NO_OMP-NOT: InlineHLFIRAssign
! NO_OMP-NOT: LowerWorkshare
! NO_OMP-NOT: LowerWorkdistribute
! NO_OMP-NOT: SimdOnlyPass
! NO_OMP-NOT: LowerNontemporalPass
! NO_OMP-NOT: MarkDeclareTargetPass
! NO_OMP-NOT: UnimplementedDeviceCheckPass
! NO_OMP-NOT: FunctionFilteringPass
! NO_OMP-NOT: HostOpFilteringPass
! NO_OMP-NOT: StackToSharedPass
! NO_OMP-NOT: PrepareForOMPOffloadPrivatizationPass

! FULL: MapsForPrivatizedSymbolsPass
! FULL: AutomapToTargetDataPass
! FULL: MapInfoFinalizationPass
! FULL: GenericLoopConversionPass
! FULL-NOT: InlineHLFIRAssign
! FULL: LowerWorkshare
! FULL: LowerWorkdistribute
! FULL-NOT: SimdOnlyPass
! FULL: LowerNontemporalPass
! FULL: MarkDeclareTargetPass
! FULL: UnimplementedDeviceCheckPass
! FULL: MarkDeclareTargetPass
! FULL: FunctionFilteringPass
! FULL: HostOpFilteringPass
! FULL: StackToSharedPass
! FULL: PrepareForOMPOffloadPrivatizationPass

! SIMD-NOT: MapsForPrivatizedSymbolsPass
! SIMD-NOT: AutomapToTargetDataPass
! SIMD-NOT: MapInfoFinalizationPass
! SIMD-NOT: GenericLoopConversionPass
! SIMD-NOT: InlineHLFIRAssign
! SIMD-NOT: LowerWorkshare
! SIMD-NOT: LowerWorkdistribute
! SIMD: SimdOnlyPass
! SIMD: LowerNontemporalPass
! SIMD-NOT: MarkDeclareTargetPass
! SIMD-NOT: UnimplementedDeviceCheckPass
! SIMD-NOT: FunctionFilteringPass
! SIMD-NOT: HostOpFilteringPass
! SIMD-NOT: StackToSharedPass
! SIMD-NOT: PrepareForOMPOffloadPrivatizationPass

! DEVICE: MapsForPrivatizedSymbolsPass
! DEVICE: AutomapToTargetDataPass
! DEVICE: MapInfoFinalizationPass
! DEVICE: GenericLoopConversionPass
! DEVICE: InlineHLFIRAssign
! DEVICE: LowerWorkshare
! DEVICE: LowerWorkdistribute
! DEVICE-NOT: SimdOnlyPass
! DEVICE: LowerNontemporalPass
! DEVICE: MarkDeclareTargetPass
! DEVICE: UnimplementedDeviceCheckPass
! DEVICE: MarkDeclareTargetPass
! DEVICE: FunctionFilteringPass
! DEVICE: HostOpFilteringPass
! DEVICE: StackToSharedPass
! DEVICE: PrepareForOMPOffloadPrivatizationPass
