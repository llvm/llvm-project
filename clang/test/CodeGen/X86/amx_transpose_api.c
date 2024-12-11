// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx512f \
// RUN: -target-feature +amx-transpose \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s --check-prefixes=CHECK

#include <immintrin.h>

char buf[2048];
#define STRIDE 32

char buf2[2048];

void test_tile_2rpntlvwz0(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test_tile_2rpntlvwz0
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz0.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz0(&dst0, &dst1, buf, STRIDE);
}

void test_tile_2rpntlvwz0t1(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test_tile_2rpntlvwz0t1
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz0t1.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz0t1(&dst0, &dst1, buf, STRIDE);
}

void test_tile_2rpntlvwz1(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test_tile_2rpntlvwz1
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz1.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz1(&dst0, &dst1, buf, STRIDE);
}

void test_tile_2rpntlvwz1t1(__tile1024i dst0, __tile1024i dst1) {
  //CHECK-LABEL: @test_tile_2rpntlvwz1t1
  //CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz1t1.internal
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 0
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  //CHECK-NEXT: {{%.*}} = extractvalue { x86_amx, x86_amx } {{%.*}}, 1
  //CHECK-NEXT: {{%.*}} = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  //CHECK-NEXT: store <256 x i32> {{%.*}}, ptr {{%.*}}
  __tile_2rpntlvwz1t1(&dst0, &dst1, buf, STRIDE);
}

void test_tile_transposed(__tile1024i dst, __tile1024i src) {
  //CHECK-LABEL: @test_tile_transposed
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call x86_amx @llvm.x86.ttransposed.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_transposed(&dst, src);
}
