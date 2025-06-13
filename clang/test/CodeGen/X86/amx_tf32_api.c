// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-tf32 -target-feature +amx-transpose  \
// RUN: -target-feature +amx-bf16 -target-feature +avx512f \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s

#include <immintrin.h>

char buf[1024];
#define STRIDE 32

char buf2[1024];

void test_tile_mmultf32ps(__tile1024i a, __tile1024i b, __tile1024i c) {
  //CHECK-LABEL: @test_tile_mmultf32ps
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call x86_amx @llvm.x86.tmmultf32ps.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_mmultf32ps(&c, a, b);
}

void test_tile_tmmultf32ps(__tile1024i a, __tile1024i b, __tile1024i c) {
  //CHECK-LABEL: @test_tile_tmmultf32ps
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call x86_amx @llvm.x86.ttmmultf32ps.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_tmmultf32ps(&c, a, b);
}
