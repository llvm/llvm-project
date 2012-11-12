// RUN: %clang_cc1 -triple=powerpc-apple-darwin8 -faltivec -fsyntax-only -verify %s

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
__vector __pixel vv_p;
__vector pixel vv__p;
__vector int vf__r();
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
vector __pixel v_p;
vector pixel v__p;
vector int f__r();
void f_a(vector int a);
void f_a2(int b, vector int a);

vector int v = (vector int)(-1);

// These should have warnings.
__vector long vv_l;                 // expected-warning {{Use of 'long' with '__vector' is deprecated}}
__vector signed long vv_sl;         // expected-warning {{Use of 'long' with '__vector' is deprecated}}
__vector unsigned long vv_ul;       // expected-warning {{Use of 'long' with '__vector' is deprecated}}
__vector long int vv_li;            // expected-warning {{Use of 'long' with '__vector' is deprecated}}
__vector signed long int vv_sli;    // expected-warning {{Use of 'long' with '__vector' is deprecated}}
__vector unsigned long int vv_uli;  // expected-warning {{Use of 'long' with '__vector' is deprecated}}
vector long v_l;                    // expected-warning {{Use of 'long' with '__vector' is deprecated}}
vector signed long v_sl;            // expected-warning {{Use of 'long' with '__vector' is deprecated}}
vector unsigned long v_ul;          // expected-warning {{Use of 'long' with '__vector' is deprecated}}
vector long int v_li;               // expected-warning {{Use of 'long' with '__vector' is deprecated}}
vector signed long int v_sli;       // expected-warning {{Use of 'long' with '__vector' is deprecated}}
vector unsigned long int v_uli;     // expected-warning {{Use of 'long' with '__vector' is deprecated}}
__vector long double  vv_ld;        // expected-warning {{Use of 'long' with '__vector' is deprecated}} expected-error {{cannot use 'double' with '__vector'}}
vector long double  v_ld;           // expected-warning {{Use of 'long' with '__vector' is deprecated}} expected-error {{cannot use 'double' with '__vector'}}
vector bool v_b;                    // expected-warning {{type specifier missing, defaults to 'int'}}

// These should have errors.
__vector double vv_d1;               // expected-error {{cannot use 'double' with '__vector'}}
vector double v_d2;                  // expected-error {{cannot use 'double' with '__vector'}}
__vector long double  vv_ld3;        // expected-warning {{Use of 'long' with '__vector' is deprecated}} expected-error {{cannot use 'double' with '__vector'}}
vector long double  v_ld4;           // expected-warning {{Use of 'long' with '__vector' is deprecated}} expected-error {{cannot use 'double' with '__vector'}}
vector bool float v_bf;              // expected-error {{cannot use 'float' with '__vector bool'}}
vector bool double v_bd;             // expected-error {{cannot use 'double' with '__vector bool'}}
vector bool pixel v_bp;              // expected-error {{cannot use '__pixel' with '__vector bool'}}
vector bool signed char v_bsc;       // expected-error {{cannot use 'signed' with '__vector bool'}}
vector bool unsigned int v_bsc2;     // expected-error {{cannot use 'unsigned' with '__vector bool'}}
vector bool long v_bl;               // expected-error {{cannot use 'long' with '__vector bool'}}
vector bool long long v_bll;         // expected-error {{cannot use 'long long' with '__vector bool'}}

typedef char i8;
typedef short i16;
typedef int i32;
struct S {
  // i8, i16, i32 here are field names, not type names.
  vector bool i8;                    // expected-error {{requires a specifier or qualifier}}
  vector pixel i16;
  vector long i32;                   // expected-warning {{deprecated}}
};

void f() {
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
  gv8 = gccv;     // expected-error {{assigning to '__attribute__((__vector_size__(2 * sizeof(unsigned int)))) unsigned int' from incompatible type '__attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int'}}
  av = gv8;       // expected-error {{assigning to '__vector unsigned int' from incompatible type '__attribute__((__vector_size__(2 * sizeof(unsigned int)))) unsigned int'}}

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
