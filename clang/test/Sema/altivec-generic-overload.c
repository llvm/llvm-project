// RUN: %clang_cc1 %s -triple=powerpc64le-unknown-linux -target-feature +altivec -target-feature +vsx -verify -verify-ignore-unexpected=note -pedantic -fsyntax-only

typedef signed char __v16sc __attribute__((__vector_size__(16)));
typedef unsigned char __v16uc __attribute__((__vector_size__(16)));
typedef signed short __v8ss __attribute__((__vector_size__(16)));
typedef unsigned short __v8us __attribute__((__vector_size__(16)));
typedef signed int __v4si __attribute__((__vector_size__(16)));
typedef unsigned int __v4ui __attribute__((__vector_size__(16)));
typedef signed long long __v2sll __attribute__((__vector_size__(16)));
typedef unsigned long long __v2ull __attribute__((__vector_size__(16)));
typedef signed __int128 __v1slll __attribute__((__vector_size__(16)));
typedef unsigned __int128 __v1ulll __attribute__((__vector_size__(16)));
typedef float __v4f __attribute__((__vector_size__(16)));
typedef double __v2d __attribute__((__vector_size__(16)));

__v16sc *__attribute__((__overloadable__)) convert1(vector signed char);
__v16uc *__attribute__((__overloadable__)) convert1(vector unsigned char);
__v8ss *__attribute__((__overloadable__)) convert1(vector signed short);
__v8us *__attribute__((__overloadable__)) convert1(vector unsigned short);
__v4si *__attribute__((__overloadable__)) convert1(vector signed int);
__v4ui *__attribute__((__overloadable__)) convert1(vector unsigned int);
__v2sll *__attribute__((__overloadable__)) convert1(vector signed long long);
__v2ull *__attribute__((__overloadable__)) convert1(vector unsigned long long);
__v1slll *__attribute__((__overloadable__)) convert1(vector signed __int128);
__v1ulll *__attribute__((__overloadable__)) convert1(vector unsigned __int128);
__v4f *__attribute__((__overloadable__)) convert1(vector float);
__v2d *__attribute__((__overloadable__)) convert1(vector double);
void __attribute__((__overloadable__)) convert1(vector bool int);
void __attribute__((__overloadable__)) convert1(vector pixel short);

vector signed char *__attribute__((__overloadable__)) convert2(__v16sc);
vector unsigned char *__attribute__((__overloadable__)) convert2(__v16uc);
vector signed short *__attribute__((__overloadable__)) convert2(__v8ss);
vector unsigned short *__attribute__((__overloadable__)) convert2(__v8us);
vector signed int *__attribute__((__overloadable__)) convert2(__v4si);
vector unsigned int *__attribute__((__overloadable__)) convert2(__v4ui);
vector signed long long *__attribute__((__overloadable__)) convert2(__v2sll);
vector unsigned long long *__attribute__((__overloadable__)) convert2(__v2ull);
vector signed __int128 *__attribute__((__overloadable__)) convert2(__v1slll);
vector unsigned __int128 *__attribute__((__overloadable__)) convert2(__v1ulll);
vector float *__attribute__((__overloadable__)) convert2(__v4f);
vector double *__attribute__((__overloadable__)) convert2(__v2d);

void test() {
  __v16sc gv1;
  __v16uc gv2;
  __v8ss gv3;
  __v8us gv4;
  __v4si gv5;
  __v4ui gv6;
  __v2sll gv7;
  __v2ull gv8;
  __v1slll gv9;
  __v1ulll gv10;
  __v4f gv11;
  __v2d gv12;

  vector signed char av1;
  vector unsigned char av2;
  vector signed short av3;
  vector unsigned short av4;
  vector signed int av5;
  vector unsigned int av6;
  vector signed long long av7;
  vector unsigned long long av8;
  vector signed __int128 av9;
  vector unsigned __int128 av10;
  vector float av11;
  vector double av12;
  vector bool int av13;
  vector pixel short av14;

  __v16sc *gv1_p = convert1(gv1);
  __v16uc *gv2_p = convert1(gv2);
  __v8ss *gv3_p = convert1(gv3);
  __v8us *gv4_p = convert1(gv4);
  __v4si *gv5_p = convert1(gv5);
  __v4ui *gv6_p = convert1(gv6);
  __v2sll *gv7_p = convert1(gv7);
  __v2ull *gv8_p = convert1(gv8);
  __v1slll *gv9_p = convert1(gv9);
  __v1ulll *gv10_p = convert1(gv10);
  __v4f *gv11_p = convert1(gv11);
  __v2d *gv12_p = convert1(gv12);

  vector signed char *av1_p = convert2(av1);
  vector unsigned char *av2_p = convert2(av2);
  vector signed short *av3_p = convert2(av3);
  vector unsigned short *av4_p = convert2(av4);
  vector signed int *av5_p = convert2(av5);
  vector unsigned int *av6_p = convert2(av6);
  vector signed long long *av7_p = convert2(av7);
  vector unsigned long long *av8_p = convert2(av8);
  vector signed __int128 *av9_p = convert2(av9);
  vector unsigned __int128 *av10_p = convert2(av10);
  vector float *av11_p = convert2(av11);
  vector double *av12_p = convert2(av12);
  convert2(av13); // expected-error {{call to 'convert2' is ambiguous}}
  convert2(av14); // expected-error {{call to 'convert2' is ambiguous}}
}
