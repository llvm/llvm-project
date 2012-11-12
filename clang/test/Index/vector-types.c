int __attribute__((vector_size(16))) x;
typedef int __attribute__((vector_size(16))) int4_t;

// RUN: c-index-test -test-print-typekind %s | FileCheck %s
// CHECK: VarDecl=x:1:38 typekind=Vector [isPOD=1]
// CHECK: TypedefDecl=int4_t:2:46 (Definition) typekind=Typedef [canonical=Vector] [isPOD=1]
