// RUN: c-index-test -test-print-type-size %s -target x86_64-pc-linux-gnu | FileCheck %s

struct Foo {
  int size;
  // CHECK: FieldDecl=size:4:7 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=0]
  void *data[];
  // CHECK: FieldDecl=data:6:9 (Definition) [type=void *[]] [typekind=IncompleteArray] [sizeof=-2] [alignof=8] [offsetof=64]
};

struct Bar {
  int size;
  // CHECK: FieldDecl=size:11:7 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=0]
  struct {
    int dummy;
    // CHECK: FieldDecl=dummy:14:9 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=64/0]
    void *data[];
    // CHECK: FieldDecl=data:16:11 (Definition) [type=void *[]] [typekind=IncompleteArray] [sizeof=-2] [alignof=8] [offsetof=128/64]
  };
};

struct Baz {
  int size;
  // CHECK: FieldDecl=size:22:7 (Definition) [type=int] [typekind=Int] [sizeof=4] [alignof=4] [offsetof=0]
  union {
    void *data1[];
    // CHECK: FieldDecl=data1:25:11 (Definition) [type=void *[]] [typekind=IncompleteArray] [sizeof=-2] [alignof=8] [offsetof=64/0]
    void *data2[];
    // CHECK: FieldDecl=data2:27:11 (Definition) [type=void *[]] [typekind=IncompleteArray] [sizeof=-2] [alignof=8] [offsetof=64/0]
  };
};

