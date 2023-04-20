// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

// RUN: clang-scan-deps -compilation-database %t/cdb.json -print-timing > %t/result.json 2>%t/errs
// RUN: cat %t/errs | FileCheck %s
// CHECK: clang-scan-deps timing: {{[0-9]+}}.{{[0-9][0-9]}}s wall, {{[0-9]+}}.{{[0-9][0-9]}}s process

//--- cdb.json
[]
