struct _MyS {
  int foo;
} MyS;

struct _MyS ww;

int x, y;

// RUN: c-index-test -cursor-at=%s:1:9 \
// RUN:              -cursor-at=%s:2:9 \
// RUN:              -cursor-at=%s:5:9 \
// RUN:              -cursor-at=%s:7:5 \
// RUN:              -cursor-at=%s:7:8 \
// RUN:       %s | FileCheck %s

// CHECK: StructDecl=_MyS:1:8 (Definition)
// CHECK: FieldDecl=foo:2:7 (Definition)
// CHECK: TypeRef=struct _MyS:1:8
// CHECK: VarDecl=x:7:5
// CHECK: VarDecl=y:7:8
