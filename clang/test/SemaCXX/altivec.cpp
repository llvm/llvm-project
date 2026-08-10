// RUN: %clang_cc1 -target-feature +altivec -flax-vector-conversions=none -triple powerpc-unknown-unknown -fcxx-exceptions -verify=expected,precxx17 %std_cxx98-14 %s
// RUN: %clang_cc1 -target-feature +altivec -flax-vector-conversions=none -triple powerpc-unknown-unknown -fcxx-exceptions -verify %std_cxx17- %s

typedef int V4i __attribute__((vector_size(16)));

void test_vec_step(vector short arg1) {
  vector bool char vbc;
  vector signed char vsc;
  vector unsigned char vuc;
  vector bool short vbs;
  vector short vs;
  vector unsigned short vus;
  vector pixel vp;
  vector bool int vbi;
  vector int vi;
  vector unsigned int vui;
  vector float vf;

  vector int *pvi;

  int res1[vec_step(arg1) == 8 ? 1 : -1];
  int res2[vec_step(vbc) == 16 ? 1 : -1];
  int res3[vec_step(vsc) == 16 ? 1 : -1];
  int res4[vec_step(vuc) == 16 ? 1 : -1];
  int res5[vec_step(vbs) == 8 ? 1 : -1];
  int res6[vec_step(vs) == 8 ? 1 : -1];
  int res7[vec_step(vus) == 8 ? 1 : -1];
  int res8[vec_step(vp) == 8 ? 1 : -1];
  int res9[vec_step(vbi) == 4 ? 1 : -1];
  int res10[vec_step(vi) == 4 ? 1 : -1];
  int res11[vec_step(vui) == 4 ? 1 : -1];
  int res12[vec_step(vf) == 4 ? 1 : -1];
  int res13[vec_step(*pvi) == 4 ? 1 : -1];
}

void f(V4i a)
{
}

void test1()
{
  V4i vGCC;
  vector int vAltiVec;

  f(vAltiVec);
  vGCC = vAltiVec;
  bool res = vGCC > vAltiVec;
  vAltiVec = 0 ? vGCC : vGCC;
}

template<typename T>
void template_f(T param) {
  param++;
}

void test2()
{
  vector int vi;
  ++vi;
  vi++;
  --vi;
  vi--;
  vector float vf;
  vf++;

  ++vi=vi; // precxx17-warning {{unsequenced}}
  (++vi)[1]=1;
  template_f(vi);
}

namespace LValueToRValueConversions {
  struct Struct {
    float f();
    int n();
  };

  vector float initFloat = (vector float)(Struct().f); // expected-error {{did you mean to call it}}
  vector int initInt = (vector int)(Struct().n); // expected-error {{did you mean to call it}}
}

void f() {
  try {}
  catch (vector pixel px) {}
};
