// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm < %s| FileCheck %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -fexperimental-new-constant-interpreter < %s| FileCheck %s

typedef __WCHAR_TYPE__ wchar_t;
typedef __SIZE_TYPE__ size_t;

void *memcpy(void *, void const *, size_t);
void *memccpy(void *, void const *, int, size_t);

// CHECK: @test1
// CHECK: call void @llvm.memset.p0.i32
// CHECK: call void @llvm.memset.p0.i32
// CHECK: call void @llvm.memcpy.p0.p0.i32
// CHECK: call void @llvm.memmove.p0.p0.i32
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

// CHECK: @test2
// CHECK: call void @llvm.memcpy.p0.p0.i32
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

// Make sure we don't over-estimate the alignment of fields of
// packed structs.
struct PS {
  int modes[4];
} __attribute__((packed));
struct PS ps;
void test8(int *arg) {
  // CHECK: @test8
  // CHECK: call void @llvm.memcpy{{.*}} align 4 {{.*}} align 1 {{.*}} 16, i1 false)
  __builtin_memcpy(arg, ps.modes, sizeof(struct PS));
}

__attribute((aligned(16))) int x[4], y[4];
void test9(void) {
  // CHECK: @test9
  // CHECK: call void @llvm.memcpy{{.*}} align 16 {{.*}} align 16 {{.*}} 16, i1 false)
  __builtin_memcpy(x, y, sizeof(y));
}

wchar_t dest;
wchar_t src;

// CHECK-LABEL: @test10
// FIXME: Consider lowering these to llvm.memcpy / llvm.memmove.
void test10(void) {
  // CHECK: call ptr @wmemcpy(ptr noundef @dest, ptr noundef @src, i32 noundef 4)
  __builtin_wmemcpy(&dest, &src, 4);

  // CHECK: call ptr @wmemmove(ptr noundef @dest, ptr noundef @src, i32 noundef 4)
  __builtin_wmemmove(&dest, &src, 4);
}

// CHECK-LABEL: @test11
void test11(void) {
  typedef struct { int a; } b;
  int d;
  b e;
  // CHECK: call void @llvm.memcpy{{.*}}(
  memcpy(&d, (char *)&e.a, sizeof(e));
}

// CHECK-LABEL: @test12
extern char dest_array[];
extern char src_array[];
void test12(void) {
  // CHECK: call void @llvm.memcpy{{.*}}(
  memcpy(&dest_array, &dest_array, 2);
}

// CHECK-LABEL: @test13
void test13(char *d, char *s, int c, size_t n) {
  // CHECK: call ptr @memccpy
  memccpy(d, s, c, n);
}
