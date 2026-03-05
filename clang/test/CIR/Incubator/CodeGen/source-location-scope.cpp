// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir

// Test for crash when getLoc() is called with invalid SourceLocation
// and currSrcLoc is not set. This can happen with compiler-generated
// expressions like CXXDefaultArgExpr and CXXDefaultInitExpr.

//===----------------------------------------------------------------------===//
// CXXDefaultArgExpr - default argument expressions
//===----------------------------------------------------------------------===//

struct S {
  int x;
  int y;
};

// Default argument expressions can have invalid source locations
void foo(S s = {}) {
  S local = s;
}

void testDefaultArg() {
  foo();
}

//===----------------------------------------------------------------------===//
// CXXDefaultInitExpr - default member initializers
//===----------------------------------------------------------------------===//

struct Inner {
  int value;
};

struct Outer {
  Inner inner = {};  // Default member initializer
  int x = 42;
};

void testDefaultInit() {
  Outer o;
  (void)o;
}

// CIR-DAG: cir.func {{.*}}@_Z3foo1S
// CIR-DAG: cir.func {{.*}}@_Z14testDefaultArgv
// CIR-DAG: cir.func {{.*}}@_Z15testDefaultInitv

// Verify that CXXDefaultArgExpr gets proper source locations from the
// default argument's initializer expression, not unknown locations.
// The struct initialization should have a fused location covering the struct
// definition (lines 12-15 where "struct S { int x; int y; };" is).
// CIR-DAG: #[[LOC_S_START:loc[0-9]*]] = loc({{.*}}:12:1)
// CIR-DAG: #[[LOC_S_END:loc[0-9]*]] = loc({{.*}}:15:1)
// CIR-DAG: #[[LOC_FUSED:loc[0-9]*]] = loc(fused[#[[LOC_S_START]], #[[LOC_S_END]]])
// CIR-DAG: cir.store {{.*}} loc(#[[LOC_FUSED]])

// Verify that CXXDefaultInitExpr gets proper source locations from the
// default member initializer expression, not unknown locations.
// The constant 42 should have a location pointing to line 36 where "int x = 42" is.
// CIR-DAG: #[[LOC42:loc[0-9]*]] = loc({{.*}}:36:11)
// CIR-DAG: cir.const #cir.int<42> : !s32i loc(#[[LOC42]])
