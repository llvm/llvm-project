// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-avx512 -target-feature +avx10.2-512 \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s --check-prefixes=CHECK

#include <immintrin.h>

char buf[1024];
#define STRIDE 32

char buf2[1024];

__m512 test_tile_cvtrowd2ps(__tile1024i a, unsigned b) {
  //CHECK-LABEL: @test_tile_cvtrowd2ps
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call <16 x float> @llvm.x86.tcvtrowd2ps.internal
 return __tile_cvtrowd2ps(a, b);
}

__m512bh test_tile_cvtrowps2pbf16h(__tile1024i a, unsigned b) {
  //CHECK-LABEL: @test_tile_cvtrowps2pbf16h
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call <32 x bfloat> @llvm.x86.tcvtrowps2pbf16h.internal
 return __tile_cvtrowps2pbf16h(a, b);
}

__m512bh test_tile_cvtrowps2pbf16l(__tile1024i a, unsigned b) {
  //CHECK-LABEL: @test_tile_cvtrowps2pbf16l
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call <32 x bfloat> @llvm.x86.tcvtrowps2pbf16l.internal
 return __tile_cvtrowps2pbf16l(a, b);
}

__m512h test_tile_cvtrowps2phh(__tile1024i a, unsigned b) {
  //CHECK-LABEL: @test_tile_cvtrowps2phh
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call <32 x half> @llvm.x86.tcvtrowps2phh.internal
 return __tile_cvtrowps2phh(a, b);
}

__m512h test_tile_cvtrowps2phl(__tile1024i a, unsigned b) {
  //CHECK-LABEL: @test_tile_cvtrowps2phl
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call <32 x half> @llvm.x86.tcvtrowps2phl.internal
 return __tile_cvtrowps2phl(a, b);
}

__m512i test_tile_movrow(__tile1024i a, unsigned b) {
  //CHECK-LABEL: @test_tile_movrow
  //CHECK-DAG: call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> {{%.*}})
  //CHECK-DAG: call <16 x i32> @llvm.x86.tilemovrow.internal
 return __tile_movrow(a, b);
}
