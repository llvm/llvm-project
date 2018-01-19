// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm < %s| FileCheck %s

// CHECK: @test1
// CHECK: call void @llvm.memset.p0i8.i32
// CHECK: call void @llvm.memset.p0i8.i32
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
// CHECK: call void @llvm.memmove.p0i8.p0i8.i32
// CHECK-NOT: __builtin
// CHECK: ret
int test1(int argc, char **argv) {
  unsigned char a = 0x11223344;
  unsigned char b = 0x11223344;
  __builtin_bzero(&a, sizeof(a));
  __builtin_memset(&a, 0, sizeof(a));
  __builtin_memcpy(&a, &b, sizeof(a));
  __builtin_memmove(&a, &b, sizeof(a));
  return 0;
}

// rdar://9289468

// CHECK: @test2
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i32
char* test2(char* a, char* b) {
  return __builtin_memcpy(a, b, 4);
}

// CHECK: @test3
// CHECK: call void @llvm.memset
void test3(char *P) {
  __builtin___memset_chk(P, 42, 128, 128);
}

// CHECK: @test4
// CHECK: call void @llvm.memcpy
void test4(char *P, char *Q) {
  __builtin___memcpy_chk(P, Q, 128, 128);
}

// CHECK: @test5
// CHECK: call void @llvm.memmove
void test5(char *P, char *Q) {
  __builtin___memmove_chk(P, Q, 128, 128);
}

// CHECK: @test6
// CHECK: call void @llvm.memcpy
int test6(char *X) {
  return __builtin___memcpy_chk(X, X, 42, 42) != 0;
}

// CHECK: @test7
// PR12094
int test7(int *p) {
  struct snd_pcm_hw_params_t* hwparams;  // incomplete type.
  
  // CHECK: call void @llvm.memset{{.*}} align 4 {{.*}}256, i1 false)
  __builtin_memset(p, 0, 256);  // Should be alignment = 4

  // CHECK: call void @llvm.memset{{.*}} align 1 {{.*}}256, i1 false)
  __builtin_memset((char*)p, 0, 256);  // Should be alignment = 1

  __builtin_memset(hwparams, 0, 256);  // No crash alignment = 1
  // CHECK: call void @llvm.memset{{.*}} align 1{{.*}}256, i1 false)
}

// <rdar://problem/11314941>
// Make sure we don't over-estimate the alignment of fields of
// packed structs.
struct PS {
  int modes[4];
} __attribute__((packed));
struct PS ps;
void test8(int *arg) {
  // CHECK: @test8
  // CHECK: call void @llvm.memcpy{{.*}} align 1 {{.*}} align 1 {{.*}} 16, i1 false)
  __builtin_memcpy(arg, ps.modes, sizeof(struct PS));
}

__attribute((aligned(16))) int x[4], y[4];
void test9() {
  // CHECK: @test9
  // CHECK: call void @llvm.memcpy{{.*}} align 16 {{.*}} align 16 {{.*}} 16, i1 false)
  __builtin_memcpy(x, y, sizeof(y));
}
