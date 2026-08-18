// Verify that --mlir-print-ir-before and --mlir-print-ir-after work in fir-opt
// for FIR/HLFIR-specific passes registered by registerFlangPipelinePasses().

// RUN: fir-opt --convert-hlfir-to-fir --mlir-print-ir-before=convert-hlfir-to-fir %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BEFORE
// RUN: fir-opt --convert-hlfir-to-fir --mlir-print-ir-after=convert-hlfir-to-fir %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=AFTER

// BEFORE: IR Dump Before ConvertHLFIRtoFIR
// BEFORE: func.func @dummy()

// AFTER: IR Dump After ConvertHLFIRtoFIR
// AFTER: func.func @dummy()

func.func @dummy() {
  return
}
