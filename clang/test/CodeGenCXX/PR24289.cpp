// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm %s -o - -triple=x86_64-pc-linux-gnu -std=c++11 | FileCheck %s

namespace std {
template <class T>
struct initializer_list {
  const T *Begin;
  __SIZE_TYPE__ Size;

  constexpr initializer_list(const T *B, __SIZE_TYPE__ S)
      : Begin(B), Size(S) {}
};
}

void f() {
  static std::initializer_list<std::initializer_list<int>> a{
      {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
  static std::initializer_list<std::initializer_list<int>> b{
      {0}, {0}, {0}, {0}};
  static std::initializer_list<std::initializer_list<int>> c{
      {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
  static std::initializer_list<std::initializer_list<int>> d{
      {0}, {0}, {0}, {0}, {0}};
  static std::initializer_list<std::initializer_list<int>> e{
      {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
}

// CHECK-DAG: @_ZZ1fvE1a = internal global %{{.*}} { %{{.*}}* getelementptr inbounds ([14 x %{{.*}}], [14 x %{{.*}}]
// CHECK-DAG: * @_ZGRZ1fvE1a_, i32 0, i32 0), i64 14 }
// CHECK-DAG: @_ZGRZ1fvE1a0_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a1_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a2_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a3_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a4_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a5_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a6_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a7_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a8_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a9_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1aA_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1aB_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1aC_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1aD_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1a_ = internal constant [14 x %{{.*}}] [%{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a0_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a1_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a2_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a3_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a4_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a5_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a6_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a7_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a8_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1a9_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1aA_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1aB_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1aC_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1aD_, i32 0, i32 0), i64 1 }]
// CHECK-DAG: @_ZZ1fvE1b = internal global %{{.*}} { %{{.*}}* getelementptr inbounds ([4 x %{{.*}}], [4 x %{{.*}}]*
// CHECK-DAG: @_ZGRZ1fvE1b_, i32 0, i32 0), i64 4 }
// CHECK-DAG: @_ZGRZ1fvE1b0_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1b1_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1b2_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1b3_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1b_ = internal constant [4 x %{{.*}}] [%{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1b0_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1b1_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1b2_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1b3_, i32 0, i32 0), i64 1 }]
// CHECK-DAG: @_ZZ1fvE1c = internal global %{{.*}} { %{{.*}}* getelementptr inbounds ([9 x %{{.*}}], [9 x %{{.*}}]*
// CHECK-DAG: @_ZGRZ1fvE1c_, i32 0, i32 0), i64 9 }
// CHECK-DAG: @_ZGRZ1fvE1c0_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c1_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c2_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c3_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c4_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c5_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c6_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c7_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c8_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1c_ = internal constant [9 x %{{.*}}] [%{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c0_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c1_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c2_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c3_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c4_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c5_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c6_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c7_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1c8_, i32 0, i32 0), i64 1 }]
// CHECK-DAG: @_ZZ1fvE1d = internal global %{{.*}} { %{{.*}}* getelementptr inbounds ([5 x %{{.*}}], [5 x %{{.*}}]* @_ZGRZ1fvE1d_, i32 0, i32 0), i64 5 }
// CHECK-DAG: @_ZGRZ1fvE1d0_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1d1_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1d2_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1d3_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1d4_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1d_ = internal constant [5 x %{{.*}}] [%{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1d0_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1d1_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1d2_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1d3_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1d4_, i32 0, i32 0), i64 1 }]
// CHECK-DAG: @_ZZ1fvE1e = internal global %{{.*}} { %{{.*}}* getelementptr inbounds ([11 x %{{.*}}], [11 x %{{.*}}]* @_ZGRZ1fvE1e_, i32 0, i32 0), i64 11 }
// CHECK-DAG: @_ZGRZ1fvE1e0_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e1_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e2_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e3_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e4_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e5_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e6_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e7_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e8_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e9_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1eA_ = internal constant [1 x i32] zeroinitializer
// CHECK-DAG: @_ZGRZ1fvE1e_ = internal constant [11 x %{{.*}}] [%{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e0_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e1_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e2_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e3_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e4_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e5_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e6_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e7_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e8_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1e9_, i32 0, i32 0), i64 1 }, %{{.*}} { i32* getelementptr inbounds ([1 x i32], [1 x i32]* @_ZGRZ1fvE1eA_, i32 0, i32 0), i64 1 }]
