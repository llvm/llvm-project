// RUN: %clang_cc1 -fms-extensions -w -triple i386-pc-win32 -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define dso_local i64 @f1_1()
// CHECK-LABEL: define dso_local void @f1_2(i32 %a0.0, i32 %a0.1)
struct s1 {
  int a;
  int b;
};
struct s1 f1_1(void) { while (1) {} }
void f1_2(struct s1 a0) {}

// CHECK-LABEL: define dso_local i32 @f2_1()
struct s2 {
  short a;
  short b;
};
struct s2 f2_1(void) { while (1) {} }

// CHECK-LABEL: define dso_local i16 @f3_1()
struct s3 {
  char a;
  char b;
};
struct s3 f3_1(void) { while (1) {} }

// CHECK-LABEL: define dso_local i8 @f4_1()
struct s4 {
  char a:4;
  char b:4;
};
struct s4 f4_1(void) { while (1) {} }

// CHECK-LABEL: define dso_local i64 @f5_1()
// CHECK-LABEL: define dso_local void @f5_2(double %a0.0)
struct s5 {
  double a;
};
struct s5 f5_1(void) { while (1) {} }
void f5_2(struct s5 a0) {}

// CHECK-LABEL: define dso_local i32 @f6_1()
// CHECK-LABEL: define dso_local void @f6_2(float %a0.0)
struct s6 {
  float a;
};
struct s6 f6_1(void) { while (1) {} }
void f6_2(struct s6 a0) {}


// MSVC passes up to three vectors in registers, and the rest indirectly.  We
// (arbitrarily) pass oversized vectors indirectly, since that is the safest way
// to do it.
typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
typedef float __m256 __attribute__((__vector_size__(32), __aligned__(32)));
typedef float __m512 __attribute__((__vector_size__(64), __aligned__(64)));
typedef float __m1024 __attribute__((__vector_size__(128), __aligned__(128)));

__m128 gv128;
__m256 gv256;
__m512 gv512;
__m1024 gv1024;

void receive_vec_128(__m128 x, __m128 y, __m128 z, __m128 w, __m128 q) {
  gv128 = x + y + z + w + q;
}
void receive_vec_256(__m256 x, __m256 y, __m256 z, __m256 w, __m256 q) {
  gv256 = x + y + z + w + q;
}
void receive_vec_512(__m512 x, __m512 y, __m512 z, __m512 w, __m512 q) {
  gv512 = x + y + z + w + q;
}
void receive_vec_1024(__m1024 x, __m1024 y, __m1024 z, __m1024 w, __m1024 q) {
  gv1024 = x + y + z + w + q;
}
// CHECK-LABEL: define dso_local void @receive_vec_128(<4 x float> inreg noundef %x, <4 x float> inreg noundef %y, <4 x float> inreg noundef %z, ptr noundef %0, ptr noundef %1)
// CHECK-LABEL: define dso_local void @receive_vec_256(<8 x float> inreg noundef %x, <8 x float> inreg noundef %y, <8 x float> inreg noundef %z, ptr noundef %0, ptr noundef %1)
// CHECK-LABEL: define dso_local void @receive_vec_512(<16 x float> inreg noundef %x, <16 x float> inreg noundef %y, <16 x float> inreg noundef %z, ptr noundef %0, ptr noundef %1)
// CHECK-LABEL: define dso_local void @receive_vec_1024(ptr noundef %0, ptr noundef %1, ptr noundef %2, ptr noundef %3, ptr noundef %4)

void pass_vec_128(void) {
  __m128 z = {0};
  receive_vec_128(z, z, z, z, z);
}

// CHECK-LABEL: define dso_local void @pass_vec_128()
// CHECK: call void @receive_vec_128(<4 x float> inreg noundef %{{[^,)]*}}, <4 x float> inreg noundef %{{[^,)]*}}, <4 x float> inreg noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}}, ptr noundef %{{[^,)]*}})


void __fastcall fastcall_indirect_vec(__m128 x, __m128 y, __m128 z, __m128 w, int edx, __m128 q) {
  gv128 = x + y + z + w + q;
}
// CHECK-LABEL: define dso_local x86_fastcallcc void @"\01@fastcall_indirect_vec@84"(<4 x float> inreg noundef %x, <4 x float> inreg noundef %y, <4 x float> inreg noundef %z, ptr inreg noundef %0, i32 inreg noundef %edx, ptr noundef %1)

struct __declspec(align(1)) Align1 { unsigned long long x; };
struct __declspec(align(4)) Align4 { unsigned long long x; };
struct __declspec(align(8)) Align8 { unsigned long long x; };
void receive_align1(struct Align1 o);
void receive_align4(struct Align4 o);
void receive_align8(struct Align8 o);
void pass_underaligned_record() {
  struct Align1 a1;
  receive_align1(a1);
  struct Align4 a4;
  receive_align4(a4);
  struct Align8 a8;
  receive_align8(a8);
}
// CHECK-LABEL: define dso_local void @pass_underaligned_record()
// CHECK: call void @receive_align1(i64 {{[^,)]*}})
// CHECK: call void @receive_align4(i64 {{[^,)]*}})
// CHECK: call void @receive_align8(ptr {{[^,)]*}})

struct FieldAlign1 { unsigned long long __declspec(align(1)) x; };
struct FieldAlign4 { unsigned long long __declspec(align(4)) x; };
struct FieldAlign8 { unsigned long long __declspec(align(8)) x; };
void receive_falign1(struct FieldAlign1 o);
void receive_falign4(struct FieldAlign4 o);
void receive_falign8(struct FieldAlign8 o);
void pass_underaligned_record_field() {
  struct FieldAlign1 a1;
  receive_falign1(a1);
  struct FieldAlign4 a4;
  receive_falign4(a4);
  struct FieldAlign8 a8;
  receive_falign8(a8);
}
// CHECK-LABEL: define dso_local void @pass_underaligned_record_field()
// CHECK: call void @receive_falign1(i64 {{[^,)]*}})
// CHECK: call void @receive_falign4(i64 {{[^,)]*}})
// CHECK: call void @receive_falign8(ptr {{[^,)]*}})
