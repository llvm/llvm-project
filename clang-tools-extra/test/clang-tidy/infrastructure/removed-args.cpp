// RUN: not clang-tidy %s --allow-no-checks -- -fnot-an-option | FileCheck %s -check-prefix=INVALID-A
// RUN: clang-tidy %s --allow-no-checks --config="{RemovedArgs: ['-fnot-an-option']}" -- -fnot-an-option
// RUN: clang-tidy %s --allow-no-checks --config="{RemovedArgs: ['-fnot-another-option', '-fnot-an-option']}" -- -fnot-an-option -fnot-another-option
// RUN: clang-tidy %s --allow-no-checks --removed-arg="-fnot-an-option" -- -fnot-an-option
// RUN: not clang-tidy %s --allow-no-checks --removed-arg="-fnot-an-option" -- -fnot-an-option -fnot-another-option | FileCheck %s -check-prefix=INVALID-B

// INVALID-A: error: unknown argument: '-fnot-an-option' [clang-diagnostic-error]
// INVALID-B: error: unknown argument: '-fnot-another-option' [clang-diagnostic-error]
