// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s -stack-protector 1 | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s | FileCheck %s --check-prefix=NOMARK
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s -stack-protector 2 | FileCheck %s --check-prefix=NOMARK
// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -o - %s -stack-protector 3 | FileCheck %s --check-prefix=NOMARK

// NOMARK-NOT: !stack-protector-padding

void use(void *);

// clang adds [12 x i8] tail padding to the IR type.
struct alignas(16) Padded {
  int a;
};
// CHECK-DAG: %p = alloca %struct.Padded, align 16, !stack-protector-padding ![[PADDED:[0-9]+]]{{$}}
// CHECK-DAG: ![[PADDED]] = !{i64 16, i64 4, i64 12}
void padded() {
  Padded p;
  use(&p);
}

// Same IR type as Padded, but the [12 x i8] is an actual field in the source code.
struct alignas(16) PaddedBuf {
  int a;
  char buf[12];
};
// CHECK-DAG: %p = alloca %struct.PaddedBuf, align 16, !stack-protector-padding ![[PADDEDBUF:[0-9]+]]{{$}}
// CHECK-DAG: ![[PADDEDBUF]] = !{i64 16}
void paddedBuf() {
  PaddedBuf p;
  use(&p);
}

// Has padding, but also a buffer, which the stack protector still sees.
struct alignas(32) BufAndPadding {
  char buf[16];
};
// CHECK-DAG: %b = alloca %struct.BufAndPadding, align 32, !stack-protector-padding ![[BUFANDPADDING:[0-9]+]]{{$}}
// CHECK-DAG: ![[BUFANDPADDING]] = !{i64 32, i64 16, i64 16}
void bufAndPadding() {
  BufAndPadding b;
  use(&b);
}

// The vptr is not padding.
struct alignas(32) Dynamic {
  virtual void f();
  int a;
};
// CHECK-DAG: %d = alloca %struct.Dynamic, align 32, !stack-protector-padding ![[DYNAMIC:[0-9]+]]{{$}}
// CHECK-DAG: ![[DYNAMIC]] = !{i64 32, i64 12, i64 20}
void dynamic() {
  Dynamic d;
  use(&d);
}

struct alignas(16) NestedPaddedInner {
  int a;
};
struct NestedPadded {
  NestedPaddedInner p;
  int x;
};
// CHECK-DAG: %n = alloca %struct.NestedPadded, align 16, !stack-protector-padding ![[NESTEDPADDED:[0-9]+]]{{$}}
// CHECK-DAG: ![[NESTEDPADDED]] = !{i64 32, i64 4, i64 12, i64 20, i64 12}
void nestedPadded() {
  NestedPadded n;
  use(&n);
}

struct alignas(8) SmallPadding {
  int a;
};
// CHECK-DAG: %s = alloca %struct.SmallPadding, align 8, !stack-protector-padding ![[SMALLPADDING:[0-9]+]]{{$}}
// CHECK-DAG: ![[SMALLPADDING]] = !{i64 8, i64 4, i64 4}
void smallPadding() {
  SmallPadding s;
  use(&s);
}

struct alignas(16) SmallBufPadded {
  int a;
  char buf[4];
};
// CHECK-DAG: %s = alloca %struct.SmallBufPadded, align 16, !stack-protector-padding ![[SMALLBUFPADDED:[0-9]+]]{{$}}
// CHECK-DAG: ![[SMALLBUFPADDED]] = !{i64 16, i64 8, i64 8}
void smallBufPadded() {
  SmallBufPadded s;
  use(&s);
}

// Only whole bytes of padding are listed.
struct alignas(16) BitField {
  unsigned a : 4;
};
// CHECK-DAG: %b = alloca %struct.BitField, align 16, !stack-protector-padding ![[BITFIELD:[0-9]+]]{{$}}
// CHECK-DAG: ![[BITFIELD]] = !{i64 16, i64 1, i64 15}
void bitField() {
  BitField b;
  use(&b);
}

// Unnamed bit-fields are padding.
struct UnnamedBitFields {
  int a;
  int : 32;
  int : 32;
  int b;
};
// CHECK-DAG: %u = alloca %struct.UnnamedBitFields, align 4, !stack-protector-padding ![[UNNAMEDBITFIELDS:[0-9]+]]{{$}}
// CHECK-DAG: ![[UNNAMEDBITFIELDS]] = !{i64 16, i64 4, i64 8}
void unnamedBitFields() {
  UnnamedBitFields u;
  use(&u);
}

// Padding within a single byte, bits [4, 8), is no whole byte of padding, so
// only the size is listed.
struct SubBytePadding {
  char a : 4;
  char b;
};
// CHECK-DAG: %s = alloca %struct.SubBytePadding, align 1, !stack-protector-padding ![[SUBBYTEPADDING:[0-9]+]]{{$}}
// CHECK-DAG: ![[SUBBYTEPADDING]] = !{i64 2}
void subBytePadding() {
  SubBytePadding s;
  use(&s);
}

// Likewise for bits [2, 5).
struct SubBytePaddingInside {
  char a : 2;
  char : 3;
  char b : 3;
};
// CHECK-DAG: %s = alloca %struct.SubBytePaddingInside, align 1, !stack-protector-padding ![[SUBBYTEPADDINGINSIDE:[0-9]+]]{{$}}
// CHECK-DAG: ![[SUBBYTEPADDINGINSIDE]] = !{i64 1}
void subBytePaddingInside() {
  SubBytePaddingInside s;
  use(&s);
}

struct BigBuf {
  int a;
  char buf[8];
};
// CHECK-DAG: %b = alloca %struct.BigBuf, align 4, !stack-protector-padding ![[BIGBUF:[0-9]+]]{{$}}
// CHECK-DAG: ![[BIGBUF]] = !{i64 12}
void bigBuf() {
  BigBuf b;
  use(&b);
}

struct alignas(16) NestedBufPadded {
  int a;
};
struct NestedBufBuf {
  int a;
  char buf[8];
};
struct NestedBuf {
  NestedBufPadded p;
  NestedBufBuf b;
};
// CHECK-DAG: %n = alloca %struct.NestedBuf, align 16, !stack-protector-padding ![[NESTEDBUF:[0-9]+]]{{$}}
// CHECK-DAG: ![[NESTEDBUF]] = !{i64 32, i64 4, i64 12, i64 28, i64 4}
void nestedBuf() {
  NestedBuf n;
  use(&n);
}

union IntUnion {
  int i;
  double d;
};
// CHECK-DAG: %u = alloca %union.IntUnion, align 8, !stack-protector-padding ![[INTUNION:[0-9]+]]{{$}}
// CHECK-DAG: ![[INTUNION]] = !{i64 8}
void intUnion() {
  IntUnion u;
  use(&u);
}

union BufUnion {
  int i;
  char buf[20];
};
// CHECK-DAG: %u = alloca %union.BufUnion, align 4, !stack-protector-padding ![[BUFUNION:[0-9]+]]{{$}}
// CHECK-DAG: ![[BUFUNION]] = !{i64 20}
void bufUnion() {
  BufUnion u;
  use(&u);
}

// Only records get the metadata.
struct alignas(16) ArrayElement {
  int a;
};
// CHECK-DAG: %i = alloca i32, align 4{{$}}
// CHECK-DAG: %c = alloca [16 x i8], align 1{{$}}
// CHECK-DAG: %p = alloca [4 x %struct.ArrayElement], align 16{{$}}
void nonRecords() {
  int i;
  char c[16];
  ArrayElement p[4];
  use(&i);
  use(c);
  use(p);
}

struct alignas(64) IgnoredPadded {
  int a;
};
// CHECK-DAG: %p = alloca %struct.IgnoredPadded, align 64, !stack-protector-padding ![[IGNOREDPADDED:[0-9]+]], !stack-protector ![[IGN:[0-9]+]]{{$}}
// CHECK-DAG: ![[IGNOREDPADDED]] = !{i64 64, i64 4, i64 60}
// CHECK-DAG: ![[IGN]] = !{i32 0}
void ignored() {
  __attribute__((stack_protector_ignore)) IgnoredPadded p;
  use(&p);
}
