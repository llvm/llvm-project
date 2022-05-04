// RUN: %clang_cc1 -triple=powerpc-apple-darwin8 -target-feature +altivec -fsyntax-only -verify=expected,novsx %s
// RUN: %clang_cc1 -triple=powerpc64-unknown-linux-gnu -target-feature +altivec -target-feature +vsx -fsyntax-only -verify=expected,nonaix %s
// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu -target-feature +altivec -target-feature -vsx -fsyntax-only -verify=expected,novsx %s
// RUN: %clang_cc1 -triple=powerpc-ibm-aix -target-feature +altivec -fsyntax-only -verify=expected,aix %s
// fRUN: %clang_cc1 -triple=powerpc64-ibm-aix -target-feature +altivec -target-feature -vsx -fsyntax-only -verify=expected,aix %s

__vector char vv_c;
__vector signed char vv_sc;
__vector unsigned char vv_uc;
__vector short vv_s;
__vector signed  short vv_ss;
__vector unsigned  short vv_us;
__vector short int vv_si;
__vector signed short int vv_ssi;
__vector unsigned short int vv_usi;
__vector int vv_i;
__vector signed int vv_sint;
__vector unsigned int vv_ui;
__vector float vv_f;
__vector bool char vv_bc;
__vector bool short vv_bs;
__vector bool int vv_bi;
__vector __bool char vv___bc;
__vector __bool short vv___bs;
__vector __bool int vv___bi;
__vector __pixel vv_p;
__vector pixel vv__p;
__vector int vf__r(void);
void vf__a(__vector int a);
void vf__a2(int b, __vector int a);

vector char v_c;
vector signed char v_sc;
vector unsigned char v_uc;
vector short v_s;
vector signed  short v_ss;
vector unsigned  short v_us;
vector short int v_si;
vector signed short int v_ssi;
vector unsigned short int v_usi;
vector int v_i;
vector signed int v_sint;
vector unsigned int v_ui;
vector float v_f;
vector bool char v_bc;
vector bool short v_bs;
vector bool int v_bi;
vector __bool char v___bc;
vector __bool short v___bs;
vector __bool int v___bi;
vector __pixel v_p;
vector pixel v__p;
vector int f__r(void);
void f_a(vector int a);
void f_a2(int b, vector int a);

vector int v = (vector int)(-1);

// These should have errors on AIX and warnings otherwise.
__vector long vv_l;                 // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
__vector signed long vv_sl;         // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
__vector unsigned long vv_ul;       // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
__vector long int vv_li;            // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
__vector signed long int vv_sli;    // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
__vector unsigned long int vv_uli;  // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
vector long v_l;                    // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
vector signed long v_sl;            // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
vector unsigned long v_ul;          // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
vector long int v_li;               // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
vector signed long int v_sli;       // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}
vector unsigned long int v_uli;     // nonaix-warning {{Use of 'long' with '__vector' is deprecated}}
                                    // aix-error@-1 {{cannot use 'long' with '__vector'}}
                                    // novsx-error@-2 {{cannot use 'long' with '__vector'}}

// These should have warnings.
__vector long double  vv_ld;        // expected-error {{cannot use 'long double' with '__vector'}}
vector long double  v_ld;           // expected-error {{cannot use 'long double' with '__vector'}}
vector bool v_b;                    // expected-error {{type specifier missing, defaults to 'int'}}
vector __bool v___b;                // expected-error {{type specifier missing, defaults to 'int'}}

// These should have errors.
#ifndef __VSX__
__vector double vv_d1;               // expected-error {{use of 'double' with '__vector' requires VSX support to be enabled (available on POWER7 or later)}}
vector double v_d2;                  // expected-error {{use of 'double' with '__vector' requires VSX support to be enabled (available on POWER7 or later)}}
__vector bool long long v_bll1;      // expected-error {{use of 'long long' with '__vector' requires VSX support (available on POWER7 or later) to be enabled}}
__vector __bool long long v_bll2;    // expected-error {{use of 'long long' with '__vector' requires VSX support (available on POWER7 or later) to be enabled}}
vector bool long long v_bll3;        // expected-error {{use of 'long long' with '__vector' requires VSX support (available on POWER7 or later) to be enabled}}
vector __bool long long v_bll4;      // expected-error {{use of 'long long' with '__vector' requires VSX support (available on POWER7 or later) to be enabled}}
#endif
__vector long double  vv_ld3;        // expected-error {{cannot use 'long double' with '__vector'}}
vector long double  v_ld4;           // expected-error {{cannot use 'long double' with '__vector'}}
vector bool float v_bf;              // expected-error {{cannot use 'float' with '__vector bool'}}
vector bool double v_bd;             // expected-error {{cannot use 'double' with '__vector bool'}}
vector bool pixel v_bp;              // expected-error {{cannot use '__pixel' with '__vector bool'}}
vector bool signed char v_bsc;       // expected-error {{cannot use 'signed' with '__vector bool'}}
vector bool unsigned int v_bsc2;     // expected-error {{cannot use 'unsigned' with '__vector bool'}}
vector bool long v_bl;               // expected-error {{cannot use 'long' with '__vector bool'}}
vector __bool float v___bf;          // expected-error {{cannot use 'float' with '__vector bool'}}
vector __bool double v___bd;         // expected-error {{cannot use 'double' with '__vector bool'}}
vector __bool pixel v___bp;          // expected-error {{cannot use '__pixel' with '__vector bool'}}
vector __bool signed char v___bsc;   // expected-error {{cannot use 'signed' with '__vector bool'}}
vector __bool unsigned int v___bsc2; // expected-error {{cannot use 'unsigned' with '__vector bool'}}
vector __bool long v___bl;           // expected-error {{cannot use 'long' with '__vector bool'}}

#ifdef __VSX__
// vector long is deprecated, but vector long long is not.
vector long long v_ll;
vector signed long long v_sll;
vector unsigned long long v_ull;
#else
// vector long long is not supported without vsx.
vector long long v_ll;              //  expected-error {{use of 'long long' with '__vector' requires VSX support (available on POWER7 or later) to be enabled}}
vector signed long long v_sll;      //  expected-error {{use of 'long long' with '__vector' requires VSX support (available on POWER7 or later) to be enabled}}
vector unsigned long long v_ull;    //  expected-error {{use of 'long long' with '__vector' requires VSX support (available on POWER7 or later) to be enabled}}
#endif

typedef char i8;
typedef short i16;
typedef int i32;
struct S {
  // i8, i16, i32 here are field names, not type names.
  vector bool i8;                    // expected-error {{requires a specifier or qualifier}}
  vector pixel i16;
  vector short i32;
};

void f(void) {
  __vector unsigned int v = {0,0,0,0};
  __vector int v__cast = (__vector int)v;
  __vector int v_cast = (vector int)v;
  __vector char vb_cast = (vector char)v;

  // Check some casting between gcc and altivec vectors.
  #define gccvector __attribute__((vector_size(16)))
  gccvector unsigned int gccv = {0,0,0,0};
  gccvector unsigned int gccv1 = gccv;
  gccvector int gccv2 = (gccvector int)gccv;
  gccvector unsigned int gccv3 = v;
  __vector unsigned int av = gccv;
  __vector int avi = (__vector int)gccv;
  gccvector unsigned int gv = v;
  gccvector int gvi = (gccvector int)v;
  __attribute__((vector_size(8))) unsigned int gv8;
  gv8 = gccv;     // expected-error {{assigning to '__attribute__((__vector_size__(2 * sizeof(unsigned int)))) unsigned int' (vector of 2 'unsigned int' values) from incompatible type '__attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int' (vector of 4 'unsigned int' values)}}
  av = gv8;       // expected-error {{assigning to '__vector unsigned int' (vector of 4 'unsigned int' values) from incompatible type '__attribute__((__vector_size__(2 * sizeof(unsigned int)))) unsigned int' (vector of 2 'unsigned int' values)}}

  v = gccv;
  __vector unsigned int tv = gccv;
  gccv = v;
  gccvector unsigned int tgv = v;

  int res_i;
  // bug 7553 - Problem with '==' and vectors
  res_i = (vv_sc == vv_sc);
  res_i = (vv_uc != vv_uc);
  res_i = (vv_s > vv_s);
  res_i = (vv_us >= vv_us);
  res_i = (vv_i < vv_i);
  res_i = (vv_ui <= vv_ui);
  res_i = (vv_f <= vv_f);
}

// bug 6895 - Vectorl literal casting confusion.
vector char v1 = (vector char)((vector int)(1, 2, 3, 4));
vector char v2 = (vector char)((vector float)(1.0f, 2.0f, 3.0f, 4.0f));
vector char v3 = (vector char)((vector int)('a', 'b', 'c', 'd'));
vector int v4 = (vector int)(1, 2, 3, 4);
vector float v5 = (vector float)(1.0f, 2.0f, 3.0f, 4.0f);
vector char v6 = (vector char)((vector int)(1+2, -2, (int)(2.0 * 3), -(5-3)));
