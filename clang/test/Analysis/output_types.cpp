// RUN: not %clang_analyze_cc1 %s \
// RUN:   -analyzer-output=plist \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-PLIST

// CHECK-PLIST: error: analyzer output type 'plist' requires an output file to
// CHECK-PLIST-SAME: be specified with -o </path/to/output_file>


// RUN: not %clang_analyze_cc1 %s \
// RUN:   -analyzer-output=plist-multi-file \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-PLIST-MULTI

// CHECK-PLIST-MULTI: error: analyzer output type 'plist-multi-file' requires
// CHECK-PLIST-MULTI-SAME: an output file to be specified with
// CHECK-PLIST-MULTI-SAME: -o </path/to/output_file>


// RUN: not %clang_analyze_cc1 %s \
// RUN:   -analyzer-output=plist-html \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-PLIST-HTML

// CHECK-PLIST-HTML: error: analyzer output type 'plist-html' requires an output
// CHECK-PLIST-HTML-SAME: directory to be specified with
// CHECK-PLIST-HTML-SAME: -o </path/to/output_directory>


// RUN: not %clang_analyze_cc1 %s \
// RUN:   -analyzer-output=sarif \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-SARIF

// CHECK-SARIF: error: analyzer output type 'sarif' requires an output file to
// CHECK-SARIF-SAME: be specified with -o </path/to/output_file>


// RUN: not %clang_analyze_cc1 %s \
// RUN:   -analyzer-output=html \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-HTML

// CHECK-HTML: error: analyzer output type 'html' requires an output directory
// CHECK-HTML-SAME: to be specified with -o </path/to/output_directory>


// RUN: not %clang_analyze_cc1 %s \
// RUN:   -analyzer-output=html-single-file \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-HTML-SINGLE

// CHECK-HTML-SINGLE: error: analyzer output type 'html-single-file' requires
// CHECK-HTML-SINGLE-SAME: an output directory to be specified with
// CHECK-HTML-SINGLE-SAME: -o </path/to/output_directory>
