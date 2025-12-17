// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -finclude-default-header -verify %s

float2x3 gM;

void bad_index_type(float f) {
  gM[f]; // expected-error {{matrix row index is not an integer}}
}

// 2 rows: valid row indices: 0, 1
void bad_constant_row_index() {
  gM[2]; // expected-error {{matrix row index is outside the allowed range}}
}
