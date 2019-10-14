// RUN: rm -rf %t.dir
// RUN: rm -rf %t-cdb.json
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/static-analyzer.c
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/analyze_header_input.h
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/static-analyzer-cdb.json > %t-cdb.json
//
// RUN: clang-scan-deps -compilation-database %t-cdb.json -j 1 | FileCheck %s

#ifdef __clang_analyzer__
#include "Inputs/analyze_header_input.h"
#endif

// CHECK: analyze_header_input.h

