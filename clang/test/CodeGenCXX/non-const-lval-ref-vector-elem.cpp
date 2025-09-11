// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-llvm %s -o - | FileCheck %s

using v4i = int __attribute__((ext_vector_type(4)));

struct v4i_s { int x, y, z, w; };

// CHECK-LABEL: define dso_local noundef i32 @_Z7f_basicv()
// CHECK: entry:
// CHECK:   %v = alloca <4 x i32>, align 16
// CHECK:   %r = alloca ptr, align 8
// CHECK:   store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %v, align 16
// CHECK:   %0 = load <4 x i32>, ptr %v, align 16
// CHECK:   %vecins = insertelement <4 x i32> %0, i32 7, i32 0
// CHECK:   store <4 x i32> %vecins, ptr %v, align 16
// CHECK:   %1 = load <4 x i32>, ptr %v, align 16
// CHECK:   %vecext = extractelement <4 x i32> %1, i32 0
// CHECK:   ret i32 %vecext
int f_basic() {
  v4i v = {1, 2, 3, 4};
  int &r = v[0];
  r = 7;
  return v[0];
}

// CHECK-LABEL: define dso_local noundef i32 @_Z8f_varidxRDv4_ii(
// CHECK: entry:
// CHECK:   %v.addr = alloca ptr, align 8
// CHECK:   %i.addr = alloca i32, align 4
// CHECK:   %r = alloca ptr, align 8
// CHECK:   store ptr %v, ptr %v.addr, align 8
// CHECK:   store i32 %i, ptr %i.addr, align 4
// CHECK:   %0 = load ptr, ptr %v.addr, align 8
// CHECK:   %1 = load i32, ptr %i.addr, align 4
// CHECK:   %2 = load <4 x i32>, ptr %0, align 16
// CHECK:   %vecext = extractelement <4 x i32> %2, i32 %1
// CHECK:   %add = add nsw i32 %vecext, 1
// CHECK:   %3 = load <4 x i32>, ptr %0, align 16
// CHECK:   %vecins = insertelement <4 x i32> %3, i32 %add, i32 %1
// CHECK:   store <4 x i32> %vecins, ptr %0, align 16
// CHECK:   %4 = load ptr, ptr %v.addr, align 8
// CHECK:   %5 = load <4 x i32>, ptr %4, align 16
// CHECK:   %6 = load i32, ptr %i.addr, align 4
// CHECK:   %vecext1 = extractelement <4 x i32> %5, i32 %6
// CHECK:   ret i32 %vecext1
int f_varidx(v4i &v, int i) {
  int &r = v[i];
  r = r + 1;
  return v[i];
}

int cast_ref_read(v4i_s &v, int i) {
  return ((v4i&)v)[i];
}

int cast_ptr_read(v4i_s *v, int i) {
  return (*((v4i*)v))[i];
}