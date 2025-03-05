// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -target-feature +v8.3a -ffreestanding -fsyntax-only -verify -verify-ignore-unexpected=note  %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

void test(float64x1_t v1f64, float64x2_t v2f64) {
    vcmla_f64(v1f64, v1f64, v1f64);  // expected-error {{call to undeclared function 'vcmla_f64'}}
    vcmla_lane_f64(v1f64, v1f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmla_lane_f64'}}
    vcmla_laneq_f64(v1f64, v1f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmla_laneq_f64'}}
    vcmlaq_lane_f64(v2f64, v2f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmlaq_lane_f64'}}
    vcmlaq_laneq_f64(v2f64, v2f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmlaq_laneq_f64'}}

    vcmla_rot90_f64(v1f64, v1f64, v1f64); // expected-error {{call to undeclared function 'vcmla_rot90_f64'}}
    vcmla_rot90_lane_f64(v1f64, v1f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmla_rot90_lane_f64'}}
    vcmla_rot90_laneq_f64(v1f64, v1f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmla_rot90_laneq_f64'}}
    vcmlaq_rot90_lane_f64(v2f64, v2f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmlaq_rot90_lane_f64'}}
    vcmlaq_rot90_laneq_f64(v2f64, v2f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmlaq_rot90_laneq_f64'}}

    vcmla_rot180_f64(v1f64, v1f64, v1f64); // expected-error {{call to undeclared function 'vcmla_rot180_f64'}}
    vcmla_rot180_lane_f64(v1f64, v1f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmla_rot180_lane_f64'}}
    vcmla_rot180_laneq_f64(v1f64, v1f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmla_rot180_laneq_f64'}}
    vcmlaq_rot180_lane_f64(v2f64, v2f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmlaq_rot180_lane_f64'}}
    vcmlaq_rot180_laneq_f64(v2f64, v2f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmlaq_rot180_laneq_f64'}}

    vcmla_rot270_f64(v1f64, v1f64, v1f64); // expected-error {{call to undeclared function 'vcmla_rot270_f64'}}
    vcmla_rot270_lane_f64(v1f64, v1f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmla_rot270_lane_f64'}}
    vcmla_rot270_laneq_f64(v1f64, v1f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmla_rot270_laneq_f64'}}
    vcmlaq_rot270_lane_f64(v2f64, v2f64, v1f64, 0); // expected-error {{call to undeclared function 'vcmlaq_rot270_lane_f64'}}
    vcmlaq_rot270_laneq_f64(v1f64, v1f64, v2f64, 0); // expected-error {{call to undeclared function 'vcmlaq_rot270_laneq_f64'}}
}
