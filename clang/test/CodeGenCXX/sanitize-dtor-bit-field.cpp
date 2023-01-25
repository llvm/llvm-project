// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"
// RUN: %clang_cc1 -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -disable-llvm-passes -std=c++11 -triple=x86_64-pc-linux -emit-llvm -debug-info-kind=line-tables-only -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

// 24 bytes total
struct Packed {
  // Packed into 4 bytes
  unsigned int a : 1;
  unsigned int b : 1;
  //unsigned int c : 1;
  // Force alignment to next 4 bytes
  unsigned int   : 0;
  unsigned int d : 1;
  // Force alignment, 8 more bytes
  double e = 5.0;
  // 4 bytes
  unsigned int f : 1;
  ~Packed() {}
};
Packed p;


// 1 byte total
struct Empty {
  unsigned int : 0;
  ~Empty() {}
};
Empty e;


// 4 byte total
struct Simple {
  unsigned int a : 1;
  ~Simple() {}
};
Simple s;


// 8 bytes total
struct Anon {
  // 1 byte
  unsigned int a : 1;
  unsigned int b : 2;
  // Force alignment to next byte
  unsigned int   : 0;
  unsigned int c : 1;
  ~Anon() {}
};
Anon an;


struct CharStruct {
  char c;
  ~CharStruct();
};

struct Adjacent {
  CharStruct a;
  int b : 1;
  CharStruct c;
  ~Adjacent() {}
};
Adjacent ad;

// CHECK-LABEL: define {{.*}}PackedD2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}i64 17{{.*}}, !dbg ![[DI1:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: define {{.*}}EmptyD2Ev
// CHECK: ret void

// CHECK-LABEL: define {{.*}}SimpleD2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}i64 1{{.*}}, !dbg ![[DI2:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: define {{.*}}AnonD2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}i64 5{{.*}}, !dbg ![[DI3:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: define {{.*}}AdjacentD2Ev
// CHECK: call void @__sanitizer_dtor_callback_fields({{.*}}i64 1{{.*}}, !dbg ![[DI4:[0-9]+]]
// CHECK: ret void

// CHECK-LABEL: !DIFile{{.*}}cpp

// CHECK-DAG: ![[DI1]] = {{.*}}line: [[@LINE-78]]
// CHECK-DAG: ![[DI2]] = {{.*}}line: [[@LINE-54]]
// CHECK-DAG: ![[DI3]] = {{.*}}line: [[@LINE-46]]
// CHECK-DAG: ![[DI4]] = {{.*}}line: [[@LINE-30]]
