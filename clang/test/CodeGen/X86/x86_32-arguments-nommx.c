// RUN: %clang_cc1 -target-feature -mmx -target-feature +sse2 -triple i686-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

// no-mmx should put mmx into memory
typedef int __attribute__((vector_size (8))) i32v2;
int a(i32v2 x) { return x[0]; }
// CHECK-LABEL: define{{.*}} i32 @a(i64 noundef %x.coerce)

// but SSE2 vectors should still go into an SSE2 register 
typedef int __attribute__((vector_size (16))) i32v4;
int b(i32v4 x) { return x[0]; }
// CHECK-LABEL: define{{.*}} i32 @b(<4 x i32> noundef %x)
