! Verify that registerAIIRContextCLOptions, registerPassManagerCLOptions and
! registerAsmPrinterCLOptions `-maiir` options  are available to the driver.

! RUN: %flang_fc1  -maiir --help | FileCheck %s --check-prefix=AIIR

! AIIR: flang (AIIR option parsing) [options]
! Registered via registerPassManagerCLOptions
! AIIR: --aiir-pass-pipeline-local-reproducer
! Registered via registerAsmPrinterCLOptions
! AIIR: --aiir-print-local-scope
! Registered via registerAIIRContextCLOptions
! AIIR: --aiir-print-op-on-diagnostic
