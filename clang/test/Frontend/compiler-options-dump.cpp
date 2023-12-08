// RUN: %clang_cc1 -compiler-options-dump -std=c++03 %s -o - | FileCheck %s --check-prefix=CXX03
// RUN: %clang_cc1 -compiler-options-dump -std=c++17 %s -o - | FileCheck %s --check-prefix=CXX17
// RUN: %clang_cc1 -compiler-options-dump -std=c99 -x c %s -o - | FileCheck %s --check-prefix=C99

// CXX03: "features"
// CXX03: "cxx_auto_type" : false
// CXX03: "cxx_range_for" : false
// CXX03: "extensions"
// CXX03: "cxx_range_for" : true

// CXX17: "features"
// CXX17: "cxx_auto_type" : true
// CXX17: "cxx_range_for" : true
// CXX17: "extensions"
// CXX17: "cxx_range_for" : true

// C99: "features"
// C99: "c_alignas" : false
// C99: "c_atomic" : false
// C99: "cxx_auto_type" : false
// C99: "cxx_range_for" : false
// C99: "extensions"
// C99: "c_alignas" : true
// C99: "c_atomic" : true
// C99: "cxx_range_for" : false
