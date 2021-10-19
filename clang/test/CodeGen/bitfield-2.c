// RUN: %clang_cc1 -emit-llvm -triple x86_64 -O3 -o %t.opt.ll %s \
// RUN:   -fdump-record-layouts > %t.dump.txt
// RUN: FileCheck -check-prefix=CHECK-RECORD < %t.dump.txt %s
// RUN: FileCheck -check-prefix=CHECK-OPT < %t.opt.ll %s

/****/

// Check that we don't read off the end a packed 24-bit structure.
// PR6176

// CHECK-RECORD: *** Dumping IRgen Record Layout
// CHECK-RECORD: Record: RecordDecl{{.*}}s0
// CHECK-RECORD: Layout: <CGRecordLayout
// CHECK-RECORD:   LLVMType:%struct.s0 = type { [3 x i8] }
// CHECK-RECORD:   IsZeroInitializable:1
// CHECK-RECORD:   BitFields:[
// CHECK-RECORD:     <CGBitFieldInfo Offset:0 Size:24 IsSigned:1 StorageSize:24 StorageOffset:0
struct __attribute((packed)) s0 {
  int f0 : 24;
};

struct s0 g0 = { 0xdeadbeef };

int f0_load(struct s0 *a0) {
  int size_check[sizeof(struct s0) == 3 ? 1 : -1];
  return a0->f0;
}
int f0_store(struct s0 *a0) {
  return (a0->f0 = 1);
}
int f0_reload(struct s0 *a0) {
  return (a0->f0 += 1);
}

// CHECK-OPT-LABEL: define{{.*}} i64 @test_0()
// CHECK-OPT:  ret i64 1
// CHECK-OPT: }
unsigned long long test_0() {
  struct s0 g0 = { 0xdeadbeef };
  unsigned long long res = 0;
  res ^= g0.f0;
  res ^= f0_load(&g0) ^ f0_store(&g0) ^ f0_reload(&g0);
  res ^= g0.f0;
  return res;
}

/****/

// PR5591

// CHECK-RECORD: *** Dumping IRgen Record Layout
// CHECK-RECORD: Record: RecordDecl{{.*}}s1
// CHECK-RECORD: Layout: <CGRecordLayout
// CHECK-RECORD:   LLVMType:%struct.s1 = type { [3 x i8] }
// CHECK-RECORD:   IsZeroInitializable:1
// CHECK-RECORD:   BitFields:[
// CHECK-RECORD:     <CGBitFieldInfo Offset:0 Size:10 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-RECORD:     <CGBitFieldInfo Offset:10 Size:10 IsSigned:1 StorageSize:24 StorageOffset:0

#pragma pack(push)
#pragma pack(1)
struct __attribute((packed)) s1 {
  signed f0 : 10;
  signed f1 : 10;
};
#pragma pack(pop)

struct s1 g1 = { 0xdeadbeef, 0xdeadbeef };

int f1_load(struct s1 *a0) {
  int size_check[sizeof(struct s1) == 3 ? 1 : -1];
  return a0->f1;
}
int f1_store(struct s1 *a0) {
  return (a0->f1 = 1234);
}
int f1_reload(struct s1 *a0) {
  return (a0->f1 += 1234);
}

// CHECK-OPT-LABEL: define{{.*}} i64 @test_1()
// CHECK-OPT:  ret i64 210
// CHECK-OPT: }
unsigned long long test_1() {
  struct s1 g1 = { 0xdeadbeef, 0xdeadbeef };
  unsigned long long res = 0;
  res ^= g1.f0 ^ g1.f1;
  res ^= f1_load(&g1) ^ f1_store(&g1) ^ f1_reload(&g1);
  res ^= g1.f0 ^ g1.f1;
  return res;
}

/****/

// Check that we don't access beyond the bounds of a union.
//
// PR5567

// CHECK-RECORD: *** Dumping IRgen Record Layout
// CHECK-RECORD: Record: RecordDecl{{.*}}u2
// CHECK-RECORD: Layout: <CGRecordLayout
// CHECK-RECORD:   LLVMType:%union.u2 = type { i8 }
// CHECK-RECORD:   IsZeroInitializable:1
// CHECK-RECORD:   BitFields:[
// CHECK-RECORD:     <CGBitFieldInfo Offset:0 Size:3 IsSigned:0 StorageSize:8 StorageOffset:0

union __attribute__((packed)) u2 {
  unsigned long long f0 : 3;
};

union u2 g2 = { 0xdeadbeef };

int f2_load(union u2 *a0) {
  return a0->f0;
}
int f2_store(union u2 *a0) {
  return (a0->f0 = 1234);
}
int f2_reload(union u2 *a0) {
  return (a0->f0 += 1234);
}

// CHECK-OPT-LABEL: define{{.*}} i64 @test_2()
// CHECK-OPT:  ret i64 2
// CHECK-OPT: }
unsigned long long test_2() {
  union u2 g2 = { 0xdeadbeef };
  unsigned long long res = 0;
  res ^= g2.f0;
  res ^= f2_load(&g2) ^ f2_store(&g2) ^ f2_reload(&g2);
  res ^= g2.f0;
  return res;
}

/***/

// PR5039

struct s3 {
  long long f0 : 32;
  long long f1 : 32;
};

struct s3 g3 = { 0xdeadbeef, 0xdeadbeef };

int f3_load(struct s3 *a0) {
  a0->f0 = 1;
  return a0->f0;
}
int f3_store(struct s3 *a0) {
  a0->f0 = 1;
  return (a0->f0 = 1234);
}
int f3_reload(struct s3 *a0) {
  a0->f0 = 1;
  return (a0->f0 += 1234);
}

// CHECK-OPT-LABEL: define{{.*}} i64 @test_3()
// CHECK-OPT:  ret i64 -559039940
// CHECK-OPT: }
unsigned long long test_3() {
  struct s3 g3 = { 0xdeadbeef, 0xdeadbeef };
  unsigned long long res = 0;
  res ^= g3.f0 ^ g3.f1;
  res ^= f3_load(&g3) ^ f3_store(&g3) ^ f3_reload(&g3);
  res ^= g3.f0 ^ g3.f1;
  return res;
}

/***/

// This is a case where the bitfield access will straddle an alignment boundary
// of its underlying type.

struct s4 {
  unsigned f0 : 16;
  unsigned f1 : 28 __attribute__ ((packed));
};

struct s4 g4 = { 0xdeadbeef, 0xdeadbeef };

int f4_load(struct s4 *a0) {
  return a0->f0 ^ a0->f1;
}
int f4_store(struct s4 *a0) {
  return (a0->f0 = 1234) ^ (a0->f1 = 5678);
}
int f4_reload(struct s4 *a0) {
  return (a0->f0 += 1234) ^ (a0->f1 += 5678);
}

// CHECK-OPT-LABEL: define{{.*}} i64 @test_4()
// CHECK-OPT:  ret i64 4860
// CHECK-OPT: }
unsigned long long test_4() {
  struct s4 g4 = { 0xdeadbeef, 0xdeadbeef };
  unsigned long long res = 0;
  res ^= g4.f0 ^ g4.f1;
  res ^= f4_load(&g4) ^ f4_store(&g4) ^ f4_reload(&g4);
  res ^= g4.f0 ^ g4.f1;
  return res;
}

/***/

struct s5 {
  unsigned f0 : 2;
  _Bool f1 : 1;
  _Bool f2 : 1;
};

struct s5 g5 = { 0xdeadbeef, 0xdeadbeef };

int f5_load(struct s5 *a0) {
  return a0->f0 ^ a0->f1;
}
int f5_store(struct s5 *a0) {
  return (a0->f0 = 0xF) ^ (a0->f1 = 0xF) ^ (a0->f2 = 0xF);
}
int f5_reload(struct s5 *a0) {
  return (a0->f0 += 0xF) ^ (a0->f1 += 0xF) ^ (a0->f2 += 0xF);
}

// CHECK-OPT-LABEL: define{{.*}} i64 @test_5()
// CHECK-OPT:  ret i64 2
// CHECK-OPT: }
unsigned long long test_5() {
  struct s5 g5 = { 0xdeadbeef, 0xdeadbeef, 0xdeadbeef };
  unsigned long long res = 0;
  res ^= g5.f0 ^ g5.f1 ^ g5.f2;
  res ^= f5_load(&g5) ^ f5_store(&g5) ^ f5_reload(&g5);
  res ^= g5.f0 ^ g5.f1 ^ g5.f2;
  return res;
}

/***/

struct s6 {
  unsigned f0 : 2;
};

struct s6 g6 = { 0xF };

int f6_load(struct s6 *a0) {
  return a0->f0;
}
int f6_store(struct s6 *a0) {
  return a0->f0 = 0x0;
}
int f6_reload(struct s6 *a0) {
  return (a0->f0 += 0xF);
}

// CHECK-OPT-LABEL: define{{.*}} zeroext i1 @test_6()
// CHECK-OPT:  ret i1 true
// CHECK-OPT: }
_Bool test_6() {
  struct s6 g6 = { 0xF };
  unsigned long long res = 0;
  res ^= g6.f0;
  res ^= f6_load(&g6);
  res ^= g6.f0;
  return res;
}

/***/

// Check that we compute the best alignment possible for each access.
//
// CHECK-RECORD: *** Dumping IRgen Record Layout
// CHECK-RECORD: Record: RecordDecl{{.*}}s7
// CHECK-RECORD: Layout: <CGRecordLayout
// CHECK-RECORD:   LLVMType:%struct.s7 = type { i32, i32, i32, i8, i32, [12 x i8] }
// CHECK-RECORD:   IsZeroInitializable:1
// CHECK-RECORD:   BitFields:[
// CHECK-RECORD:     <CGBitFieldInfo Offset:0 Size:5 IsSigned:1 StorageSize:8 StorageOffset:12
// CHECK-RECORD:     <CGBitFieldInfo Offset:0 Size:29 IsSigned:1 StorageSize:32 StorageOffset:16

struct __attribute__((aligned(16))) s7 {
  int a, b, c;
  int f0 : 5;
  int f1 : 29;
};

int f7_load(struct s7 *a0) {
  return a0->f0;
}

/***/

// This is a case where we narrow the access width immediately.

struct __attribute__((packed)) s8 {
  char f0 : 4;
  char f1;
  int  f2 : 4;
  char f3 : 4;
};

struct s8 g8 = { 0xF };

int f8_load(struct s8 *a0) {
  return a0->f0 ^ a0 ->f2 ^ a0->f3;
}
int f8_store(struct s8 *a0) {
  return (a0->f0 = 0xFD) ^ (a0->f2 = 0xFD) ^ (a0->f3 = 0xFD);
}
int f8_reload(struct s8 *a0) {
  return (a0->f0 += 0xFD) ^ (a0->f2 += 0xFD) ^ (a0->f3 += 0xFD);
}

// CHECK-OPT-LABEL: define{{.*}} i32 @test_8()
// CHECK-OPT:  ret i32 -3
// CHECK-OPT: }
unsigned test_8() {
  struct s8 g8 = { 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef };
  unsigned long long res = 0;
  res ^= g8.f0 ^ g8.f2 ^ g8.f3;
  res ^= f8_load(&g8) ^ f8_store(&g8) ^ f8_reload(&g8);
  res ^= g8.f0 ^ g8.f2 ^ g8.f3;
  return res;
}

/***/

// This is another case where we narrow the access width immediately.
//
// <rdar://problem/7893760>

struct __attribute__((packed)) s9 {
  unsigned f0 : 7;
  unsigned f1 : 7;
  unsigned f2 : 7;
  unsigned f3 : 7;
  unsigned f4 : 7;
  unsigned f5 : 7;
  unsigned f6 : 7;
  unsigned f7 : 7;
};

int f9_load(struct s9 *a0) {
  return a0->f7;
}
