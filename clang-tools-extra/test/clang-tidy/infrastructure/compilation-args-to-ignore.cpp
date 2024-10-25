// RUN: not clang-tidy %s -- -fnot-an-option | FileCheck %s -check-prefix=INVALID-A
// RUN: clang-tidy %s --config="{CompilationArgsToRemoveRegex: ['-fnot-an-option']}" -- -fnot-an-option
// RUN: not clang-tidy %s --config="{CompilationArgsToRemoveRegex: ['-f.*']}" -- -fnot-an-option -invalid-option | FileCheck %s -check-prefix=INVALID-B
// RUN: clang-tidy %s --config="{CompilationArgsToRemoveRegex: ['-f.*', '-invalid-option']}" -- -fnot-an-option -fnot-another-option -finvalid-option -invalid-option
// RUN: not clang-tidy %s --config="{CompilationArgsToRemoveRegex: ['\$invalid-option']}" -- -finvalid-option | FileCheck %s -check-prefix=INVALID-C

// INVALID-A: error: unknown argument: '-fnot-an-option' [clang-diagnostic-error]
// INVALID-B: error: unknown argument: '-invalid-option' [clang-diagnostic-error]
// INVALID-C: error: unknown argument: '-finvalid-option' [clang-diagnostic-error]