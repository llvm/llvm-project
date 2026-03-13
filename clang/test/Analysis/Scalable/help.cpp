// RUN: %clang     --help 2>&1 | FileCheck %s
// RUN: %clang_cc1 --help 2>&1 | FileCheck %s

// CHECK:       --ssaf-extract-summaries=<summary-names>
// CHECK-NEXT:    Comma-separated list of summary names to extract
// CHECK-NEXT:  --ssaf-tu-summary-file=<path>.<format>
// CHECK-NEXT:    The output file for the extracted summaries. The extension selects which file format to use.
