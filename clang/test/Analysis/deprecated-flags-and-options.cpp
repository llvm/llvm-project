// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPRECATED-STORE
// DEPRECATED-STORE: error: unknown argument: '-analyzer-store=region'

// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-opt-analyze-nested-blocks %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPRECATED-NESTED-BLOCKS
// DEPRECATED-NESTED-BLOCKS: error: unknown argument: '-analyzer-opt-analyze-nested-blocks'

// RUN: not %clang_analyze_cc1 -analyzer-checker=core -analyzer-config consider-single-element-arrays-as-flexible-array-members=true %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=DEPRECATED-SINGLE-ELEM-FAM
// DEPRECATED-SINGLE-ELEM-FAM: error: unknown analyzer-config 'consider-single-element-arrays-as-flexible-array-members'


void empty() {}
