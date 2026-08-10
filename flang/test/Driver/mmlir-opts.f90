! Verify that registerMLIRContextCLOptions, registerPassManagerCLOptions and
! registerAsmPrinterCLOptions `-mmlir` options  are available to the driver.

! RUN: %flang_fc1  -mmlir --help | FileCheck %s --check-prefix=MLIR

! MLIR: flang (MLIR option parsing) [options]
! Registered via registerPassManagerCLOptions
! MLIR: --mlir-pass-pipeline-local-reproducer
! Registered via registerAsmPrinterCLOptions
! MLIR: --mlir-print-local-scope
! Registered via registerMLIRContextCLOptions
! MLIR: --mlir-print-op-on-diagnostic
