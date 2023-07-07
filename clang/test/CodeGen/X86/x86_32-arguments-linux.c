// RUN: %clang_cc1 -w -fblocks -triple i386-pc-linux-gnu -target-cpu pentium4 -emit-llvm -o %t %s
// RUN: FileCheck < %t %s

// CHECK-LABEL: define{{.*}} void @f56(
// CHECK: i8 noundef signext %a0, ptr noundef byval(%struct.s56_0) align 4 %a1,
// CHECK: i64 noundef %a2.coerce, ptr noundef byval(%struct.s56_1) align 4 %0,
// CHECK: <1 x double> noundef %a4, ptr noundef byval(%struct.s56_2) align 4 %1,
// CHECK: <4 x i32> noundef %a6, ptr noundef byval(%struct.s56_3) align 4 %2,
// CHECK: <2 x double> noundef %a8, ptr noundef byval(%struct.s56_4) align 4 %3,
// CHECK: <8 x i32> noundef %a10, ptr noundef byval(%struct.s56_5) align 4 %4,
// CHECK: <4 x double> noundef %a12, ptr noundef byval(%struct.s56_6) align 4 %5)

// CHECK: call void (i32, ...) @f56_0(i32 noundef 1,
// CHECK: i32 noundef %{{.*}}, ptr noundef byval(%struct.s56_0) align 4 %{{[^ ]*}},
// CHECK: i64 noundef %{{[^ ]*}}, ptr noundef byval(%struct.s56_1) align 4 %{{[^ ]*}},
// CHECK: <1 x double> noundef %{{[^ ]*}}, ptr noundef byval(%struct.s56_2) align 4 %{{[^ ]*}},
// CHECK: <4 x i32> noundef %{{[^ ]*}}, ptr noundef byval(%struct.s56_3) align 4 %{{[^ ]*}},
// CHECK: <2 x double> noundef %{{[^ ]*}}, ptr noundef byval(%struct.s56_4) align 4 %{{[^ ]*}},
// CHECK: <8 x i32> noundef %{{[^ ]*}}, ptr noundef byval(%struct.s56_5) align 4 %{{[^ ]*}},
// CHECK: <4 x double> noundef %{{[^ ]*}}, ptr noundef byval(%struct.s56_6) align 4 %{{[^ ]*}})
// CHECK: }
//
// [i386] clang misaligns long double in structures when passed byval
// clang misaligns parameters on stack
typedef int __attribute__((vector_size (8))) t56_v2i;
typedef double __attribute__((vector_size (8))) t56_v1d;
typedef int __attribute__((vector_size (16))) t56_v4i;
typedef double __attribute__((vector_size (16))) t56_v2d;
typedef int __attribute__((vector_size (32))) t56_v8i;
typedef double __attribute__((vector_size (32))) t56_v4d;

struct s56_0 { char a; };
struct s56_1 { t56_v2i a; };
struct s56_2 { t56_v1d a; };
struct s56_3 { t56_v4i a; };
struct s56_4 { t56_v2d a; };
struct s56_5 { t56_v8i a; };
struct s56_6 { t56_v4d a; };

void f56(char a0, struct s56_0 a1, 
         t56_v2i a2, struct s56_1 a3, 
         t56_v1d a4, struct s56_2 a5, 
         t56_v4i a6, struct s56_3 a7, 
         t56_v2d a8, struct s56_4 a9, 
         t56_v8i a10, struct s56_5 a11, 
         t56_v4d a12, struct s56_6 a13) {
  extern void f56_0(int x, ...);
  f56_0(1, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
        a10, a11, a12, a13);
}
