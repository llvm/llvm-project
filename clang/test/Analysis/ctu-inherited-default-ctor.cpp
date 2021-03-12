// Should not crash with '-analyzer-opt-analyze-headers' option during CTU analysis.
//
// RUN: rm -rf %t && mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-inherited-default-ctor-other.cpp.ast \
// RUN:    %S/Inputs/ctu-inherited-default-ctor-other.cpp
// RUN: echo "c:@N@clang@S@DeclContextLookupResult@SingleElementDummyList ctu-inherited-default-ctor-other.cpp.ast" \
// RUN:   > %t/ctudir/externalDefMap.txt
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-opt-analyze-headers \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -verify %s 2>&1 | FileCheck %s
//
// expected-no-diagnostics
//
// CHECK: CTU loaded AST file: ctu-inherited-default-ctor-other.cpp.ast

namespace clang {}
namespace llvm {}
namespace clang {
class DeclContextLookupResult {
  static int *const SingleElementDummyList;
};
} // namespace clang
