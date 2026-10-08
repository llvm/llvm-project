// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu future \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu future \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr11 \
// RUN:   -fsyntax-only -verify=pwr11 %s

// Made with AI

void test_aes_encrypt_paired_invalid_imm(void) {
  __vector_pair vp1, vp2;

  // Test invalid immediate values (valid range is 0-2)
  // expected-error@+2 {{argument value 3 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_aes_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes_encrypt_paired(vp1, vp2, 3);
  // expected-error@+2 {{argument value -1 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_aes_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes_encrypt_paired(vp1, vp2, -1);
}

void test_aes_encrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes_encrypt_paired(vc, vp, 0);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes_encrypt_paired(vp, vc, 0);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type 'int'}}
  // pwr11-error@+1 {{'__builtin_aes_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res3 = __builtin_aes_encrypt_paired(vp, vp, vc);
}

void test_aes128_encrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes128 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes128_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes128_encrypt_paired(vc, vp);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes128_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes128_encrypt_paired(vp, vc);
}

void test_aes192_encrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes192 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes192_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes192_encrypt_paired(vc, vp);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes192_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes192_encrypt_paired(vp, vc);
}

void test_aes256_encrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes256 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes256_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes256_encrypt_paired(vc, vp);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes256_encrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes256_encrypt_paired(vp, vc);
}

void test_aes_decrypt_paired_invalid_imm(void) {
  __vector_pair vp1, vp2;

  // Test invalid immediate values (valid range is 0-2)
  // expected-error@+2 {{argument value 3 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_aes_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes_decrypt_paired(vp1, vp2, 3);
  // expected-error@+2 {{argument value -1 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_aes_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes_decrypt_paired(vp1, vp2, -1);
}

void test_aes_decrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes_decrypt_paired(vc, vp, 0);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes_decrypt_paired(vp, vc, 0);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type 'int'}}
  // pwr11-error@+1 {{'__builtin_aes_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res3 = __builtin_aes_decrypt_paired(vp, vp, vc);
}

void test_aes128_decrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes128 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes128_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes128_decrypt_paired(vc, vp);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes128_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes128_decrypt_paired(vp, vc);
}

void test_aes192_decrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes192 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes192_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes192_decrypt_paired(vc, vp);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes192_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes192_decrypt_paired(vp, vc);
}

void test_aes256_decrypt_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes256 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes256_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes256_decrypt_paired(vc, vp);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes256_decrypt_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes256_decrypt_paired(vp, vc);
}
void test_aes_genlastkey_paired_invalid_imm(void) {
  __vector_pair vp1;

  // Test invalid immediate values (valid range is 0-2)
  // expected-error@+2 {{argument value 3 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_aes_genlastkey_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes_genlastkey_paired(vp1, 3);
  // expected-error@+2 {{argument value -1 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_aes_genlastkey_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes_genlastkey_paired(vp1, -1);
}

void test_aes_genlastkey_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes_genlastkey_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes_genlastkey_paired(vc, 0);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type 'int'}}
  // pwr11-error@+1 {{'__builtin_aes_genlastkey_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res2 = __builtin_aes_genlastkey_paired(vp, vc);
}

void test_aes128_genlastkey_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes128 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes128_genlastkey_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes128_genlastkey_paired(vc);
}

void test_aes192_genlastkey_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes192 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes192_genlastkey_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes192_genlastkey_paired(vc);
}

void test_aes256_genlastkey_paired_type_mismatch(void) {
  __vector_pair vp;
  vector unsigned char vc;

  // Test type mismatches for aes256 variant
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type '__vector_pair'}}
  // pwr11-error@+1 {{'__builtin_aes256_genlastkey_paired' needs target feature future-vector,paired-vector-memops}}
  __vector_pair res1 = __builtin_aes256_genlastkey_paired(vc);
}

void test_galois_field_mult_invalid_imm(void) {
  vector unsigned char a, b;

  // Test invalid immediate values (valid range is 0-1)
  // expected-error@+2 {{argument value 2 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult' needs target feature future-vector}}
  vector unsigned char res1 = __builtin_galois_field_mult(a, b, 2);
  // expected-error@+2 {{argument value -1 is outside the valid range}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult' needs target feature future-vector}}
  vector unsigned char res2 = __builtin_galois_field_mult(a, b, -1);
}

void test_galois_field_mult_type_mismatch(void) {
  vector unsigned char vc;
  __vector_pair vp;

  // Test type mismatches
  // expected-error@+2 {{passing '__vector_pair' to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult' needs target feature future-vector}}
  vector unsigned char res1 = __builtin_galois_field_mult(vp, vc, 0);
  // expected-error@+2 {{passing '__vector_pair' to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult' needs target feature future-vector}}
  vector unsigned char res2 = __builtin_galois_field_mult(vc, vp, 0);
  // expected-error@+2 {{passing '__vector unsigned char' (vector of 16 'unsigned char' values) to parameter of incompatible type 'int'}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult' needs target feature future-vector}}
  vector unsigned char res3 = __builtin_galois_field_mult(vc, vc, vc);
}

void test_galois_field_mult_gcm_type_mismatch(void) {
  vector unsigned char vc;
  __vector_pair vp;

  // Test type mismatches for gcm variant
  // expected-error@+2 {{passing '__vector_pair' to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult_gcm' needs target feature future-vector}}
  vector unsigned char res1 = __builtin_galois_field_mult_gcm(vp, vc);
  // expected-error@+2 {{passing '__vector_pair' to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult_gcm' needs target feature future-vector}}
  vector unsigned char res2 = __builtin_galois_field_mult_gcm(vc, vp);
}

void test_galois_field_mult_xts_type_mismatch(void) {
  vector unsigned char vc;
  __vector_pair vp;

  // Test type mismatches for xts variant
  // expected-error@+2 {{passing '__vector_pair' to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult_xts' needs target feature future-vector}}
  vector unsigned char res1 = __builtin_galois_field_mult_xts(vp, vc);
  // expected-error@+2 {{passing '__vector_pair' to parameter of incompatible type '__vector unsigned char' (vector of 16 'unsigned char' values)}}
  // pwr11-error@+1 {{'__builtin_galois_field_mult_xts' needs target feature future-vector}}
  vector unsigned char res2 = __builtin_galois_field_mult_xts(vc, vp);
}
