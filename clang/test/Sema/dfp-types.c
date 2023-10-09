// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fexperimental-decimal-floating-point -fsyntax-only -verify=expected,c %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -std=c++2c -fexperimental-decimal-floating-point -fsyntax-only -verify=expected,cxx %s

// _Decimal32, _Decimal64, and _Decimal128 are never keywords in C++.
_Decimal32 d32; // cxx-error {{unknown type name '_Decimal32'}}
_Decimal64 d64; // cxx-error {{unknown type name '_Decimal64'}}
_Decimal128 d28; // cxx-error {{unknown type name '_Decimal128'}}

// DFP types are available via the GNU mode attribute in both C and C++.
typedef float __attribute__((mode(SD))) D32;
typedef float __attribute__((mode(DD))) D64;
typedef float __attribute__((mode(TD))) D128;

// The GNU mode attribute requires a floating point base type for DFP types.
// These are ok.
long double __attribute((mode(SD))) ldamsd;
double __attribute((mode(DD))) damdd;
_Float16 __attribute((mode(SD))) f16amsd;
__bf16 __attribute((mode(SD))) bf16amsd;
__float128 __attribute((mode(TD))) f128amtd;
// These are not ok.
void __attribute((mode(SD))) vamsd; // expected-error {{type of machine mode does not match type of base type}}
int __attribute((mode(DD))) iamdd; // expected-error {{type of machine mode does not match type of base type}}
int* __attribute((mode(TD))) ipamtd; // expected-error {{mode attribute only supported for integer and floating-point types}}
float __attribute((mode(TD))) *fapmtd; // expected-error {{mode attribute only supported for integer and floating-point types}}

// DFP types may be used as vector elements, but declaration form is restricted.
float __attribute__((mode(V4SD))) famv4sd; // expected-warning {{deprecated; use the 'vector_size' attribute instead}}
float __attribute__((mode(SD))) __attribute__((vector_size(16))) famsdv16;
D64 __attribute__((vector_size(16))) d64av16;

// DFP types are not allowed as elements of complex types.
D32 _Complex d32c; // expected-error {{'_Complex type-name' is invalid}}
_Decimal32 _Complex kd32c; // c-error {{'_Complex _Decimal32' is invalid}} \
                              cxx-error {{unknown type name '_Decimal32'}}

_Static_assert(sizeof(D32) == 4);
_Static_assert(sizeof(D64) == 8);
_Static_assert(sizeof(D128) == 16);

_Static_assert(_Alignof(D32) == 4);
_Static_assert(_Alignof(D64) == 8);
_Static_assert(_Alignof(D128) == 16);

struct s {
  D32 d32;
  D64 d64;
  D128 d128;
  union {
    D32 ud32;
    D64 ud64;
    D128 ud128;
  };
};

struct bitfield {
  D32 d32 : 32;    // expected-error {{bit-field 'd32' has non-integral type}}
  D64 d64 : 64;    // expected-error {{bit-field 'd64' has non-integral type}}
  D128 d128 : 128; // expected-error {{bit-field 'd128' has non-integral type}}
};

D32 test_d32(D32 d32) {
  return d32;
}

D64 test_d64(D64 d64) {
  return d64;
}

D128 test_d128(D128 d128) {
  return d128;
}

void test_builtin_complex(D32 d32) {
  __auto_type lv = __builtin_complex(d32, d32); // expected-error {{'_Complex _Decimal32' is invalid}}
}

void test_generic(D32 d32, D64 d64, D128 d128) {
  (void)_Generic(d32,  D64 : 0, D128 : 0); // expected-error-re {{controlling expression type {{.*}} not compatible with any generic association type}}
  (void)_Generic(d64,  D32 : 0, D128 : 0); // expected-error-re {{controlling expression type {{.*}} not compatible with any generic association type}}
  (void)_Generic(d128, D32 : 0, D64  : 0); // expected-error-re {{controlling expression type {{.*}} not compatible with any generic association type}}
  _Static_assert(_Generic(d32,  D64 : 0, D128 : 0, default : 1) == 1);
  _Static_assert(_Generic(d64,  D32 : 0, D128 : 0, default : 1) == 1);
  _Static_assert(_Generic(d128, D32 : 0, D64  : 0, default : 1) == 1);
  _Static_assert(_Generic(d32,  D32 : 1, D64  : 0, D128 : 0) == 1);
  _Static_assert(_Generic(d64,  D32 : 0, D64  : 1, D128 : 0) == 1);
  _Static_assert(_Generic(d128, D32 : 0, D64  : 0, D128 : 1) == 1);
}
