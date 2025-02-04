// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +v8.3a -ffreestanding -fsyntax-only -verify %s
// REQUIRES: aarch64-registered-target

#include <arm_neon.h>
#include <arm_fp16.h>

void test_vcmla_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c){
  vcmla_lane_f16(a, b, c, 0);
  vcmla_lane_f16(a, b, c, 1);

  vcmla_lane_f16(a, b, c, 2); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_lane_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c){
  vcmla_laneq_f16(a, b, c, 0);
  vcmla_laneq_f16(a, b, c, 1);
  vcmla_laneq_f16(a, b, c, 3);

  vcmla_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_laneq_f16(a, b, c, 4);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_lane_f16(float16x8_t a, float16x8_t b, float16x4_t c){
  vcmlaq_lane_f16(a, b, c, 0);
  vcmlaq_lane_f16(a, b, c, 1);

  vcmlaq_lane_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_lane_f16(a, b, c, 2);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c){
  vcmlaq_laneq_f16(a, b, c, 0);
  vcmlaq_laneq_f16(a, b, c, 1);
  vcmlaq_laneq_f16(a, b, c, 3);

  vcmlaq_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_laneq_f16(a, b, c, 4);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_lane_f32(float32x2_t a, float32x2_t b, float32x2_t c){
  vcmla_lane_f32(a, b, c, 0);

  vcmla_lane_f32(a, b, c, 1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_lane_f32(a, b, c, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_lane_f32(a, b, c, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t c){
  vcmla_laneq_f32(a, b, c, 0);

  vcmla_laneq_f32(a, b, c, 2); // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_laneq_f32(a, b, c, -1); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t c){
  vcmlaq_laneq_f32(a, b, c, 0);
  vcmlaq_laneq_f32(a, b, c, 1);

  vcmlaq_laneq_f32(a, b, c, 2); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_laneq_f32(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot90_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c){
  vcmla_rot90_lane_f16(a, b, c, 0);
  vcmla_rot90_lane_f16(a, b, c, 1);

  vcmla_rot90_lane_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_rot90_lane_f16(a, b, c, 2); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot90_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c){
  vcmla_rot90_laneq_f16(a, b, c, 0);
  vcmla_rot90_laneq_f16(a, b, c, 3);

  vcmla_rot90_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_rot90_laneq_f16(a, b, c, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_rot90_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c){
  vcmlaq_rot90_laneq_f16(a, b, c, 0);
  vcmlaq_rot90_laneq_f16(a, b, c, 3);

  vcmlaq_rot90_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_rot90_laneq_f16(a, b, c, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot180_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c){
  vcmla_rot180_lane_f16(a, b, c, 0);
  vcmla_rot180_lane_f16(a, b, c, 1);

  vcmla_rot180_lane_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_rot180_lane_f16(a, b, c, 2); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot180_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c){
  vcmla_rot180_laneq_f16(a, b, c, 0);
  vcmla_rot180_laneq_f16(a, b, c, 3);

  vcmla_rot180_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_rot180_laneq_f16(a, b, c, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_rot180_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c){
  vcmlaq_rot180_laneq_f16(a, b, c, 0);
  vcmlaq_rot180_laneq_f16(a, b, c, 3);

  vcmlaq_rot180_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_rot180_laneq_f16(a, b, c, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot270_lane_f16(float16x4_t a, float16x4_t b, float16x4_t c){
  vcmla_rot270_lane_f16(a, b, c, 0);
  vcmla_rot270_lane_f16(a, b, c, 1);

  vcmla_rot270_lane_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_rot270_lane_f16(a, b, c, 2); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot270_laneq_f16(float16x4_t a, float16x4_t b, float16x8_t c){
  vcmla_rot270_laneq_f16(a, b, c, 0);
  vcmla_rot270_laneq_f16(a, b, c, 3);

  vcmla_rot270_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmla_rot270_laneq_f16(a, b, c, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_rot270_laneq_f16(float16x8_t a, float16x8_t b, float16x8_t c){
  vcmlaq_rot270_laneq_f16(a, b, c, 0);
  vcmlaq_rot270_laneq_f16(a, b, c, 3);

  vcmlaq_rot270_laneq_f16(a, b, c, -1); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_rot270_laneq_f16(a, b, c, 4); // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot90_lane_f32(float32x2_t a, float32x2_t b, float32x2_t c){
  vcmla_rot90_lane_f32(a, b, c, 0);

  vcmla_rot90_lane_f32(a, b, c, 1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_rot90_lane_f32(a, b, c, -1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot90_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t c){
  vcmla_rot90_laneq_f32(a, b, c, 0);
  vcmla_rot90_laneq_f32(a, b, c, 1);

  vcmla_rot90_laneq_f32(a, b, c, 2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_rot90_laneq_f32(a, b, c, -1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_rot90_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t c){
  vcmlaq_rot90_laneq_f32(a, b, c, 0);
  vcmlaq_rot90_laneq_f32(a, b, c, 1);

  vcmlaq_rot90_laneq_f32(a, b, c, 2);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_rot90_laneq_f32(a, b, c, -1);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot180_lane_f32(float32x2_t a, float32x2_t b, float32x2_t c){
  vcmla_rot180_lane_f32(a, b, c, 0);

  vcmla_rot180_lane_f32(a, b, c, 1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_rot180_lane_f32(a, b, c, -1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot180_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t c){
  vcmla_rot180_laneq_f32(a, b, c, 0);
  vcmla_rot180_laneq_f32(a, b, c, 1);

  vcmla_rot180_laneq_f32(a, b, c, 2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_rot180_laneq_f32(a, b, c, -1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_rot180_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t c){
  vcmlaq_rot90_laneq_f32(a, b, c, 0);
  vcmlaq_rot90_laneq_f32(a, b, c, 1);

  vcmlaq_rot90_laneq_f32(a, b, c, 2);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_rot90_laneq_f32(a, b, c, -1);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot270_lane_f32(float32x2_t a, float32x2_t b, float32x2_t c){
  vcmla_rot270_lane_f32(a, b, c, 0);

  vcmla_rot270_lane_f32(a, b, c, 1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_rot270_lane_f32(a, b, c, -1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmla_rot270_laneq_f32(float32x2_t a, float32x2_t b, float32x4_t c){
  vcmla_rot270_laneq_f32(a, b, c, 0);
  vcmla_rot270_laneq_f32(a, b, c, 1);

  vcmla_rot270_laneq_f32(a, b, c, 2);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vcmla_rot270_laneq_f32(a, b, c, -1);  // expected-error-re {{argument value {{.*}} is outside the valid range}}
}

void test_vcmlaq_rot270_laneq_f32(float32x4_t a, float32x4_t b, float32x4_t c){
  vcmlaq_rot270_laneq_f32(a, b, c, 0);
  vcmlaq_rot270_laneq_f32(a, b, c, 1);

  vcmlaq_rot270_laneq_f32(a, b, c, 2);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
  vcmlaq_rot270_laneq_f32(a, b, c, -1);  // expected-error-re +{{argument value {{.*}} is outside the valid range}}
}