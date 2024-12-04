// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: clang-scan-deps -compilation-database %t/cdb.json -print-timing > %t/result.json 2>%t/errs
// RUN: cat %t/errs | FileCheck %s
// CHECK:      wall time [s]              process time [s]           instruction count
// CHECK-NEXT: {{[0-9]+}}.{{([0-9]{4})}}  {{[0-9]+}}.{{([0-9]{4})}}  {{[0-9]+}}

//--- cdb.json
[]
