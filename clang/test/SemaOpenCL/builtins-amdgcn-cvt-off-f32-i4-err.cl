// RUN: %clang_cc1 -triple amdgcn-- -verify -S -o - %s

void test_builtin_amdgcn_cvt_off_f32_i4(int n) {
    struct A{ unsigned x; } a;
    __builtin_amdgcn_cvt_off_f32_i4(n, n); // expected-error {{too many arguments to function call, expected 1, have 2}}
    __builtin_amdgcn_cvt_off_f32_i4(); // expected-error {{too few arguments to function call, expected 1, have 0}}
    __builtin_amdgcn_cvt_off_f32_i4(a); // expected-error {{passing '__private struct A' to parameter of incompatible type 'int'}}
}
