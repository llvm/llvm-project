// RUN: %clang_cc1 -target-feature +altivec -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

typedef short v2i16 __attribute__((vector_size (4)));
typedef short v3i16 __attribute__((vector_size (6)));
typedef short v4i16 __attribute__((vector_size (8)));
typedef short v6i16 __attribute__((vector_size (12)));
typedef short v8i16 __attribute__((vector_size (16)));
typedef short v16i16 __attribute__((vector_size (32)));

struct v16i16 { v16i16 x; };

// CHECK: define i32 @test_v2i16(i32 %x.coerce)
v2i16 test_v2i16(v2i16 x)
{
  return x;
}

// CHECK: define i64 @test_v3i16(i64 %x.coerce)
v3i16 test_v3i16(v3i16 x)
{
  return x;
}

// CHECK: define i64 @test_v4i16(i64 %x.coerce)
v4i16 test_v4i16(v4i16 x)
{
  return x;
}

// CHECK: define <6 x i16> @test_v6i16(<6 x i16> %x)
v6i16 test_v6i16(v6i16 x)
{
  return x;
}

// CHECK: define <8 x i16> @test_v8i16(<8 x i16> %x)
v8i16 test_v8i16(v8i16 x)
{
  return x;
}

// CHECK: define void @test_v16i16(<16 x i16>* noalias sret(<16 x i16>) align 32 %agg.result, <16 x i16>* %0)
v16i16 test_v16i16(v16i16 x)
{
  return x;
}

// CHECK: define void @test_struct_v16i16(%struct.v16i16* noalias sret(%struct.v16i16) align 32 %agg.result, [2 x i128] %x.coerce)
struct v16i16 test_struct_v16i16(struct v16i16 x)
{
  return x;
}
