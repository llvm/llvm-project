// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-movrs  -emit-llvm -o - -Wall -Werror -pedantic \
// RUN: -target-feature +amx-transpose -Wno-gnu-statement-expression| FileCheck %s

#include <immintrin.h>
#include <stddef.h>

char buf[2048];
#define STRIDE 32

void test_tile_2rpntlvwz0rs(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz0rs
  // CHECK: call void @llvm.x86.t2rpntlvwz0rs(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz0rs(1, A, B);
}

void test_tile_2rpntlvwz0rst1(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz0rst1
  // CHECK: call void @llvm.x86.t2rpntlvwz0rst1(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz0rst1(1, A, B);
}

void test_tile_2rpntlvwz1rs(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz1rs
  // CHECK: call void @llvm.x86.t2rpntlvwz1rs(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz1rs(1, A, B);
}

void test_tile_2rpntlvwz1rst1(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz1rst1
  // CHECK: call void @llvm.x86.t2rpntlvwz1rst1(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz1rst1(1, A, B);
}

void test__tile_2rpntlvwz0rs(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test__tile_2rpntlvwz0rs
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz0rs.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz0rs(&dst0, &dst1, buf, STRIDE);
}

void test__tile_2rpntlvwz0rst1(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test__tile_2rpntlvwz0rst1
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz0rst1.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz0rst1(&dst0, &dst1, buf, STRIDE);
}

void test__tile_2rpntlvwz1rs(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test__tile_2rpntlvwz1rs
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz1rs.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz1rs(&dst0, &dst1, buf, STRIDE);
}

void test__tile_2rpntlvwz1rst1(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test__tile_2rpntlvwz1rst1
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz1rst1.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz1rst1(&dst0, &dst1, buf, STRIDE);
}
