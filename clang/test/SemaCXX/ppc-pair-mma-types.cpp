// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -fcxx-exceptions -target-cpu pwr10 %s -verify
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -fsyntax-only \
// RUN:   -fcxx-exceptions -target-cpu pwr10 %s -verify

// vector quad

// alias
using vq_t = __vector_quad;
void testVQAlias(int *inp, int *outp) {
  vq_t *vqin = (vq_t *)inp;
  vq_t *vqout = (vq_t *)outp;
  *vqout = *vqin;
}

class TestClassVQ {
  // method argument
public:
  void testVQArg1(__vector_quad vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_quad *vqp = (__vector_quad *)ptr;
    *vqp = vq;
    *vqp1 = vq;
  }
  void testVQArg2(const __vector_quad vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_quad *vqp = (__vector_quad *)ptr;
    *vqp = vq;
    *vqp2 = vq;
  }
  void testVQArg3(__vector_quad *vq, int *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    *vqp = *vq;
    vqp1 = vqp;
  }
  void testVQArg4(const __vector_quad *const vq, int *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    *vqp = *vq;
    vqp2 = vqp;
  }
  void testVQArg5(__vector_quad vqa[], int *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    *vqp = vqa[0];
    *vqp1 = vqa[1];
  }
  void testVQArg6(const vq_t vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_quad *vqp = (__vector_quad *)ptr;
    *vqp = vq;
    *vqp2 = vq;
  }
  void testVQArg7(const vq_t *vq, int *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    *vqp = *vq;
    vqp1 = vqp;
  }

  // method return
  __vector_quad testVQRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_quad *vqp = (__vector_quad *)ptr;
    vq1 = *vqp;
    return *vqp; // expected-error {{invalid use of PPC MMA type}}
  }

  __vector_quad *testVQRet2(int *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    vq2 = *vqp;
    return vqp + 2;
  }

  const __vector_quad *testVQRet3(int *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    vqp1 = vqp;
    return vqp + 2;
  }

  const vq_t testVQRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_quad *vqp = (__vector_quad *)ptr;
    vqp2 = vqp;
    return *vqp; // expected-error {{invalid use of PPC MMA type}}
  }

  const vq_t *testVQRet5(int *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    vq1 = *vqp;
    return vqp + 2;
  }

  // template argument
  template <typename T = __vector_quad>
  void testVQTemplate(T v, T *p) { // expected-note {{candidate template ignored: substitution failure [with T = vq_t]: invalid use of PPC MMA type}} \
                                      expected-note {{candidate template ignored: substitution failure [with T = __vector_quad]: invalid use of PPC MMA type}}
    *(p + 1) = v;
  }

  // class field
public:
  __vector_quad vq1; // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp1;

private:
  vq_t vq2; // expected-error {{invalid use of PPC MMA type}}
  vq_t *vqp2;
};

// template
template <typename T>
class ClassTemplateVQ1 {
  T t; // expected-error {{invalid use of PPC MMA type}}
};
template <typename T>
class ClassTemplateVQ2 {
  T *t;
};
template <typename T>
class ClassTemplateVQ3 {
  int foo(T t) { return 10; }
};
template <typename T, typename... Ts>
class ClassTemplateVQ4 {
public:
  T operator()(Ts...) const {} // expected-error {{invalid use of PPC MMA type}}
};
void testVQTemplate() {
  ClassTemplateVQ1<__vector_quad> t1; // expected-note {{in instantiation of template class 'ClassTemplateVQ1<__vector_quad>' requested here}}
  ClassTemplateVQ1<__vector_quad *> t2;
  ClassTemplateVQ2<__vector_quad> t3;
  ClassTemplateVQ2<__vector_quad *> t4;

  ClassTemplateVQ3<int(int, int, int)> t5;
  // The following case is not prevented but it ok, this function type cannot be
  // instantiated because we prevent any function from returning an MMA type.
  ClassTemplateVQ3<__vector_quad(int, int, int)> t6;
  ClassTemplateVQ3<int(__vector_quad, int, int)> t7; // expected-error {{invalid use of PPC MMA type}}

  ClassTemplateVQ4<int, int, int, __vector_quad> t8; // expected-note {{in instantiation of template class 'ClassTemplateVQ4<int, int, int, __vector_quad>' requested here}}
  ClassTemplateVQ4<int, int, int, __vector_quad *> t9;

  TestClassVQ tc;
  __vector_quad vq;
  __vector_quad *vqp = &vq;
  tc.testVQTemplate(&vq, &vqp);
  tc.testVQTemplate<vq_t *>(&vq, &vqp);
  tc.testVQTemplate(vq, vqp);       // expected-error {{no matching member function for call to 'testVQTemplate'}}
  tc.testVQTemplate<vq_t>(vq, vqp); // expected-error {{no matching member function for call to 'testVQTemplate'}}
}

// trailing return type
auto testVQTrailing1() {
  __vector_quad vq;
  return vq; // expected-error {{invalid use of PPC MMA type}}
}
auto testVQTrailing2() {
  __vector_quad *vqp;
  return vqp;
}
auto testVQTrailing3() -> vq_t { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad vq;
  return vq; // expected-error {{invalid use of PPC MMA type}}
}
auto testVQTrailing4() -> vq_t * {
  __vector_quad *vqp;
  return vqp;
}

// new/delete
void testVQNewDelete() {
  __vector_quad *vqp1 = new __vector_quad;
  __vector_quad *vqp2 = new __vector_quad[100];
  delete vqp1;
  delete[] vqp2;
}

// lambdas expressions
void TestVQLambda() {
  auto f1 = [](void *ptr) -> __vector_quad {
    __vector_quad *vqp = (__vector_quad *)ptr;
    return *vqp; // expected-error {{invalid use of PPC MMA type}}
  };
  auto f2 = [](void *ptr) {
    __vector_quad *vqp = (__vector_quad *)ptr;
    return *vqp; // expected-error {{invalid use of PPC MMA type}}
  };
  auto f3 = [] { __vector_quad vq; __builtin_mma_xxsetaccz(&vq); return vq; }; // expected-error {{invalid use of PPC MMA type}}
}

// cast
void TestVQCast() {
  __vector_quad vq;
  int *ip = reinterpret_cast<int *>(&vq);
  __vector_quad *vq2 = reinterpret_cast<__vector_quad *>(ip);
}

// throw
void TestVQThrow() {
  __vector_quad vq;
  throw vq; // expected-error {{invalid use of PPC MMA type}}
}

// vector pair

// alias
using vp_t = __vector_pair;
void testVPAlias(int *inp, int *outp) {
  vp_t *vpin = (vp_t *)inp;
  vp_t *vpout = (vp_t *)outp;
  *vpout = *vpin;
}

class TestClassVP {
  // method argument
public:
  void testVPArg1(__vector_pair vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_pair *vpp = (__vector_pair *)ptr;
    *vpp = vp;
    *vpp1 = vp;
  }
  void testVPArg2(const __vector_pair vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_pair *vpp = (__vector_pair *)ptr;
    *vpp = vp;
    *vpp2 = vp;
  }
  void testVPArg3(__vector_pair *vp, int *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    *vpp = *vp;
    vpp1 = vpp;
  }
  void testVPArg4(const __vector_pair *const vp, int *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    *vpp = *vp;
    vpp2 = vpp;
  }
  void testVPArg5(__vector_pair vpa[], int *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    *vpp = vpa[0];
    *vpp1 = vpa[1];
  }
  void testVPArg6(const vp_t vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_pair *vpp = (__vector_pair *)ptr;
    *vpp = vp;
    *vpp2 = vp;
  }
  void testVPArg7(const vp_t *vp, int *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    *vpp = *vp;
    vpp1 = vpp;
  }

  // method return
  __vector_pair testVPRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_pair *vpp = (__vector_pair *)ptr;
    vp1 = *vpp;
    return *vpp; // expected-error {{invalid use of PPC MMA type}}
  }

  __vector_pair *testVPRet2(int *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    vp2 = *vpp;
    return vpp + 2;
  }

  const __vector_pair *testVPRet3(int *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    vpp1 = vpp;
    return vpp + 2;
  }

  const vp_t testVPRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
    __vector_pair *vpp = (__vector_pair *)ptr;
    vpp2 = vpp;
    return *vpp; // expected-error {{invalid use of PPC MMA type}}
  }

  const vp_t *testVPRet5(int *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    vp1 = *vpp;
    return vpp + 2;
  }

  // template argument
  template <typename T = __vector_pair>
  void testVPTemplate(T v, T *p) { // expected-note {{candidate template ignored: substitution failure [with T = vp_t]: invalid use of PPC MMA type}} \
                                      expected-note {{candidate template ignored: substitution failure [with T = __vector_pair]: invalid use of PPC MMA type}}
    *(p + 1) = v;
  }

  // class field
public:
  __vector_pair vp1; // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp1;

private:
  vp_t vp2; // expected-error {{invalid use of PPC MMA type}}
  vp_t *vpp2;
};

// template
template <typename T>
class ClassTemplateVP1 {
  T t; // expected-error {{invalid use of PPC MMA type}}
};
template <typename T>
class ClassTemplateVP2 {
  T *t;
};
template <typename T>
class ClassTemplateVP3 {
  int foo(T t) { return 10; }
};
template <typename T, typename... Ts>
class ClassTemplateVP4 {
public:
  T operator()(Ts...) const {} // expected-error {{invalid use of PPC MMA type}}
};
void testVPTemplate() {
  ClassTemplateVP1<__vector_pair> t1; // expected-note {{in instantiation of template class 'ClassTemplateVP1<__vector_pair>' requested here}}
  ClassTemplateVP1<__vector_pair *> t2;
  ClassTemplateVP2<__vector_pair> t3;
  ClassTemplateVP2<__vector_pair *> t4;

  ClassTemplateVP3<int(int, int, int)> t5;
  // The following case is not prevented but it ok, this function type cannot be
  // instantiated because we prevent any function from returning an MMA type.
  ClassTemplateVP3<__vector_pair(int, int, int)> t6;
  ClassTemplateVP3<int(__vector_pair, int, int)> t7; // expected-error {{invalid use of PPC MMA type}}

  ClassTemplateVP4<int, int, int, __vector_pair> t8; // expected-note {{in instantiation of template class 'ClassTemplateVP4<int, int, int, __vector_pair>' requested here}}
  ClassTemplateVP4<int, int, int, __vector_pair *> t9;

  TestClassVP tc;
  __vector_pair vp;
  __vector_pair *vpp = &vp;
  tc.testVPTemplate(&vp, &vpp);
  tc.testVPTemplate<vp_t *>(&vp, &vpp);
  tc.testVPTemplate(vp, vpp);       // expected-error {{no matching member function for call to 'testVPTemplate'}}
  tc.testVPTemplate<vp_t>(vp, vpp); // expected-error {{no matching member function for call to 'testVPTemplate'}}
}

// trailing return type
auto testVPTrailing1() {
  __vector_pair vp;
  return vp; // expected-error {{invalid use of PPC MMA type}}
}
auto testVPTrailing2() {
  __vector_pair *vpp;
  return vpp;
}
auto testVPTrailing3() -> vp_t { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair vp;
  return vp; // expected-error {{invalid use of PPC MMA type}}
}
auto testVPTrailing4() -> vp_t * {
  __vector_pair *vpp;
  return vpp;
}

// new/delete
void testVPNewDelete() {
  __vector_pair *vpp1 = new __vector_pair;
  __vector_pair *vpp2 = new __vector_pair[100];
  delete vpp1;
  delete[] vpp2;
}

// lambdas expressions
void TestVPLambda() {
  auto f1 = [](void *ptr) -> __vector_pair {
    __vector_pair *vpp = (__vector_pair *)ptr;
    return *vpp; // expected-error {{invalid use of PPC MMA type}}
  };
  auto f2 = [](void *ptr) {
    __vector_pair *vpp = (__vector_pair *)ptr;
    return *vpp; // expected-error {{invalid use of PPC MMA type}}
  };
  auto f3 = [](vector unsigned char vc) { __vector_pair vp; __builtin_vsx_assemble_pair(&vp, vc, vc); return vp; }; // expected-error {{invalid use of PPC MMA type}}
  auto f4 = [](vector unsigned char vc) { __vector_pair vp; __builtin_vsx_build_pair(&vp, vc, vc); return vp; }; // expected-error {{invalid use of PPC MMA type}}
}

// cast
void TestVPCast() {
  __vector_pair vp;
  int *ip = reinterpret_cast<int *>(&vp);
  __vector_pair *vp2 = reinterpret_cast<__vector_pair *>(ip);
}

// throw
void TestVPThrow() {
  __vector_pair vp;
  throw vp; // expected-error {{invalid use of PPC MMA type}}
}
