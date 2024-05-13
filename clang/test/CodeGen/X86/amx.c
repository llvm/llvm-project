// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown  -target-feature +amx-int8  \
// RUN: -target-feature +amx-bf16 -emit-llvm -o - -Werror -pedantic | FileCheck %s --check-prefixes=CHECK

#include <immintrin.h>

void test_amx(void *data) {
  //CHECK-LABEL: @test_amx
  //CHECK: call void @llvm.x86.ldtilecfg(ptr %{{.*}})
  //CHECK: call void @llvm.x86.sttilecfg(ptr %{{.*}})
  //CHECK: call void @llvm.x86.tilerelease()
  //CHECK: call void @llvm.x86.tilezero(i8 3)
  //CHECK: call void @llvm.x86.tileloadd64(i8 4, ptr %{{.*}}, i64 8)
  //CHECK: call void @llvm.x86.tileloaddt164(i8 0, ptr %{{.*}}, i64 1)
  //CHECK: call void @llvm.x86.tilestored64(i8 0, ptr %{{.*}}, i64 1)
  //CHECK: call void @llvm.x86.tdpbssd(i8 1, i8 2, i8 3)
  //CHECK: call void @llvm.x86.tdpbsud(i8 1, i8 2, i8 3)
  //CHECK: call void @llvm.x86.tdpbusd(i8 1, i8 2, i8 3)
  //CHECK: call void @llvm.x86.tdpbuud(i8 1, i8 2, i8 3)
  //CHECK: call void @llvm.x86.tdpbf16ps(i8 1, i8 2, i8 3)
  _tile_loadconfig(data);
  _tile_storeconfig(data);
  _tile_release();
  _tile_zero(3);
  _tile_loadd(4, data, 8);
  _tile_stream_loadd(0, data, 1);
  _tile_stored(0, data, 1);
  _tile_dpbssd(1, 2, 3);
  _tile_dpbsud(1, 2, 3);
  _tile_dpbusd(1, 2, 3);
  _tile_dpbuud(1, 2, 3);
  _tile_dpbf16ps(1, 2, 3);
}
