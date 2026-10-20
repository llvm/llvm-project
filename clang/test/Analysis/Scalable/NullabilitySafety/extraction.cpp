// C++ contributors: evidence from a method, a constructor's member
// initializer, a lambda body, a template instantiation, and a block (attributed
// to its enclosing function) each lands in that contributor's summary.

// RUN: rm -rf %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nonnull -std=c++17 \
// RUN:   -fblocks %s --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu-1 --ssaf-tu-summary-file=%t.json
// RUN: %python %S/Inputs/decode-summary.py %t.json | FileCheck %s

void sink(int *p);
struct M { int *f; M(); void m(); };
void M::m() { static int v; sink(&v); }
M::M() : f(nullptr) {}
void with_lambda() { [] { sink(nullptr); }(); }
template <class T> T *tmpl() { static T v; return &v; }
int *inst() { return tmpl<int>(); }
void with_block() { ^{ static int v; sink(&v); }(); }

// CHECK-NOT:  {{.}}
// CHECK:      c:@F@inst# AllReturnsNonnull c:@F@inst# return
// CHECK-NEXT: c:@F@inst# NonnullEvidence c:@F@inst# return
// CHECK-NEXT: c:@F@tmpl<#I># AllReturnsNonnull c:@F@tmpl<#I># return
// CHECK-NEXT: c:@F@tmpl<#I># NonnullEvidence c:@F@tmpl<#I># return
// CHECK-NEXT: c:@F@with_block# NonnullEvidence c:@F@sink#*I# param 1
// CHECK-NEXT: c:@S@M@F@M# NullableEvidence c:@S@M@FI@f
// CHECK-NEXT: c:@S@M@F@m# NonnullEvidence c:@F@sink#*I# param 1
// CHECK-NEXT: c:{{.*}}@F@with_lambda#@Sa@F@operator()#1 NullableEvidence c:@F@sink#*I# param 1
// CHECK-NOT:  {{.}}
