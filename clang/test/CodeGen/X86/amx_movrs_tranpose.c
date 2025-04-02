// RUN:  %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN:  -target-feature +amx-movrs  -emit-llvm -o - -Wall -Werror -pedantic \
// RUN:  -target-feature +amx-transpose -Wno-gnu-statement-expression| FileCheck %s

#include <immintrin.h>
#include <stddef.h>

char buf[2048];
#define STRIDE 32

// CHECK-LABEL:  define dso_local void @test_tile_2rpntlvwz0rs_internal(
// CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz0rs.internal(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, ptr %{{.*}}, i64 %{{.*}})
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 0
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
// CHECK: store <256 x i32> %{{.*}}, ptr %{{.*}}, align 1024
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 1
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
void test_tile_2rpntlvwz0rs_internal(int row, int col0, int col1, void *D0, void *D1, void *B) {
  _tile_2rpntlvwz0rs_internal(row, col0, col1, D0, D1, B, 1);
}

// CHECK-LABEL:  define dso_local void @test_tile_2rpntlvwz0rst1_internal(
// CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz0rst1.internal(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, ptr %{{.*}}, i64 %{{.*}})
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 0
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
// CHECK: store <256 x i32> %{{.*}}, ptr %{{.*}}, align 1024
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 1
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
void test_tile_2rpntlvwz0rst1_internal(int row, int col0, int col1, void *D0, void *D1, void *B) {
  _tile_2rpntlvwz0rst1_internal(row, col0, col1, D0, D1, B, 1);
}

// CHECK-LABEL:  define dso_local void @test_tile_2rpntlvwz1rs_internal(
// CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz1rs.internal(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, ptr %{{.*}}, i64 %{{.*}})
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 0
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
// CHECK: store <256 x i32> %{{.*}}, ptr %{{.*}}, align 1024
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 1
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
void test_tile_2rpntlvwz1rs_internal(int row, int col0, int col1, void *D0, void *D1, void *B) {
  _tile_2rpntlvwz1rs_internal(row, col0, col1, D0, D1, B, 1);
}

// CHECK-LABEL:  define dso_local void @test_tile_2rpntlvwz1rst1_internal(
// CHECK: call { x86_amx, x86_amx } @llvm.x86.t2rpntlvwz1rst1.internal(i16 %{{.*}}, i16 %{{.*}}, i16 %{{.*}}, ptr %{{.*}}, i64 %{{.*}})
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 0
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
// CHECK: store <256 x i32> %{{.*}}, ptr %{{.*}}, align 1024
// CHECK: extractvalue { x86_amx, x86_amx } %{{.*}}, 1
// CHECK: call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %{{.*}})
void test_tile_2rpntlvwz1rst1_internal(int row, int col0, int col1, void *D0, void *D1, void *B) {
  _tile_2rpntlvwz1rst1_internal(row, col0, col1, D0, D1, B, 1);
}
