// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown  -target-feature +amx-fp8  \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s
#include <immintrin.h>

void test_tdpbf8ps(__tile1024i src1, __tile1024i src2, __tile1024i dst) {
  //CHECK-LABEL: @test_tdpbf8ps
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call x86_amx @llvm.x86.tdpbf8ps.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_dpbf8ps(&dst, src1, src2);
}

void test_tdpbhf8ps(__tile1024i src1, __tile1024i src2, __tile1024i dst) {
  //CHECK-LABEL: @test_tdpbhf8ps
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call x86_amx @llvm.x86.tdpbhf8ps.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_dpbhf8ps(&dst, src1, src2);
}

void test_tdphbf8ps(__tile1024i src1, __tile1024i src2, __tile1024i dst) {
  //CHECK-LABEL: @test_tdphbf8ps
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call x86_amx @llvm.x86.tdphbf8ps.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_dphbf8ps(&dst, src1, src2);
}

void test_tdphf8ps(__tile1024i src1, __tile1024i src2, __tile1024i dst) {
  //CHECK-LABEL: @test_tdphf8ps
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call x86_amx @llvm.x86.tdphf8ps.internal
  //CHECK-DAG: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx {{%.*}})
  __tile_dphf8ps(&dst, src1, src2);
}

