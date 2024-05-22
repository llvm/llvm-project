// REQUIRES: native
// RUN: c-index-test -write-pch %t.pfx.h.pch -x objective-c-header %S/Inputs/cindex-from-source.h
// RUN: c-index-test -test-load-source local %s -include %t.pfx.h > %t
// RUN: FileCheck %s < %t
// CHECK: cindex-from-source.m:{{.*}}:{{.*}}: StructDecl=s0:{{.*}}:{{.*}}
// CHECK: cindex-from-source.m:{{.*}}:{{.*}}: VarDecl=g0:{{.*}}:{{.*}}
// CHECK: cindex-from-source.m:9:1: TypeRef=t0:1:13 Extent=[9:1 - 9:3]
struct s0 {};
t0 g0;

// RUN: c-index-test -test-load-source-reparse 5 local %s -include %t.pfx.h > %t
// RUN: FileCheck %s < %t
