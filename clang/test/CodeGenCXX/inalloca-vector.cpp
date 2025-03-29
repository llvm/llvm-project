// RUN: %clang_cc1 -w -triple i686-pc-win32 -emit-llvm -o - %s | FileCheck %s

// PR44395
// MSVC passes up to three vectors in registers, and the rest indirectly. Check
// that both are compatible with an inalloca prototype.

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &o);
  unsigned handle;
};

typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
__m128 gv128;

// nt, w, and q will be in the inalloca pack.
void receive_vec_128(NonTrivial nt, __m128 x, __m128 y, __m128 z, __m128 w, __m128 q) {
  gv128 = x + y + z + w + q;
}
// CHECK-LABEL: define dso_local void  @"?receive_vec_128@@YAXUNonTrivial@@T__m128@@1111@Z"
// CHECK-SAME: (<4 x float> inreg noundef %x,
// CHECK-SAME: <4 x float> inreg noundef %y,
// CHECK-SAME: <4 x float> inreg noundef %z,
// CHECK-SAME: ptr inalloca(<{ %struct.NonTrivial, ptr, ptr }>) %0)

void pass_vec_128() {
  __m128 z = {0};
  receive_vec_128(NonTrivial(), z, z, z, z, z);
}
// CHECK-LABEL: define dso_local void @"?pass_vec_128@@YAXXZ"()
// CHECK: getelementptr inbounds nuw <{ %struct.NonTrivial, ptr, ptr }>, ptr %{{[^,]*}}, i32 0, i32 0
// CHECK: call x86_thiscallcc noundef ptr @"??0NonTrivial@@QAE@XZ"(ptr {{[^,]*}} %{{.*}})

// Store q, store temp alloca.
// CHECK: store <4 x float> %{{[^,]*}}, ptr %{{[^,]*}}, align 16
// CHECK: getelementptr inbounds nuw <{ %struct.NonTrivial, ptr, ptr }>, ptr %{{[^,]*}}, i32 0, i32 1
// CHECK: store ptr %{{[^,]*}}, ptr %{{[^,]*}}, align 4

// Store w, store temp alloca.
// CHECK: store <4 x float> %{{[^,]*}}, ptr %{{[^,]*}}, align 16
// CHECK: getelementptr inbounds nuw <{ %struct.NonTrivial, ptr, ptr }>, ptr %{{[^,]*}}, i32 0, i32 2
// CHECK: store ptr %{{[^,]*}}, ptr %{{[^,]*}}, align 4

// CHECK: call void @"?receive_vec_128@@YAXUNonTrivial@@T__m128@@1111@Z"
// CHECK-SAME: (<4 x float> inreg noundef %{{[^,]*}},
// CHECK-SAME: <4 x float> inreg noundef %{{[^,]*}},
// CHECK-SAME: <4 x float> inreg noundef %{{[^,]*}},
// CHECK-SAME: ptr inalloca(<{ %struct.NonTrivial, ptr, ptr }>) %{{[^,]*}})

// w will be passed indirectly by register, and q will be passed indirectly, but
// the pointer will be in memory.
void __fastcall fastcall_receive_vec(__m128 x, __m128 y, __m128 z, __m128 w, int edx, __m128 q, NonTrivial nt) {
  gv128 = x + y + z + w + q;
}
// CHECK-LABEL: define dso_local x86_fastcallcc void @"?fastcall_receive_vec@@Y{{[^"]*}}"
// CHECK-SAME: (<4 x float> inreg noundef %x,
// CHECK-SAME: <4 x float> inreg noundef %y,
// CHECK-SAME: <4 x float> inreg noundef %z,
// CHECK-SAME: ptr inreg noundef %0,
// CHECK-SAME: i32 inreg noundef %edx,
// CHECK-SAME: ptr inalloca(<{ ptr, %struct.NonTrivial }>) %1)


void __vectorcall vectorcall_receive_vec(double xmm0, double xmm1, double xmm2,
                                         __m128 x, __m128 y, __m128 z,
                                         __m128 w, int edx, __m128 q, NonTrivial nt) {
  gv128 = x + y + z + w + q;
}
// CHECK-LABEL: define dso_local x86_vectorcallcc void @"?vectorcall_receive_vec@@Y{{[^"]*}}"
// CHECK-SAME: (double inreg noundef %xmm0,
// CHECK-SAME: double inreg noundef %xmm1,
// CHECK-SAME: double inreg noundef %xmm2,
// CHECK-SAME: <4 x float> inreg noundef %x,
// CHECK-SAME: <4 x float> inreg noundef %y,
// CHECK-SAME: <4 x float> inreg noundef %z,
// CHECK-SAME: ptr inreg noundef %0,
// CHECK-SAME: i32 inreg noundef %edx,
// CHECK-SAME: ptr inalloca(<{ ptr, %struct.NonTrivial }>) %1)
