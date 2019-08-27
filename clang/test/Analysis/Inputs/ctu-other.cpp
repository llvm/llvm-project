#include "../ctu-hdr.h"

int callback_to_main(int x);
int f(int x) {
  return x - 1;
}

int g(int x) {
  return callback_to_main(x) + 1;
}

int h_chain(int);

int h(int x) {
  return 2 * h_chain(x);
}

namespace myns {
int fns(int x) {
  return x + 7;
}

namespace embed_ns {
int fens(int x) {
  return x - 3;
}
} // namespace embed_ns

class embed_cls {
public:
  int fecl(int x);
};
int embed_cls::fecl(int x) {
  return x - 7;
}
} // namespace myns

class mycls {
public:
  int fcl(int x);
  virtual int fvcl(int x);
  static int fscl(int x);

  class embed_cls2 {
  public:
    int fecl2(int x);
  };
};

int mycls::fcl(int x) {
  return x + 5;
}
int mycls::fvcl(int x) {
  return x + 7;
}
int mycls::fscl(int x) {
  return x + 6;
}
int mycls::embed_cls2::fecl2(int x) {
  return x - 11;
}

class derived : public mycls {
public:
  virtual int fvcl(int x) override;
};

int derived::fvcl(int x) {
  return x + 8;
}

namespace chns {
int chf2(int x);

class chcls {
public:
  int chf4(int x);
};

int chf3(int x) {
  return chcls().chf4(x);
}

int chf1(int x) {
  return chf2(x);
}
}

typedef struct { int n; } Anonymous;
int fun_using_anon_struct(int n) { Anonymous anon; anon.n = n; return anon.n; }

int other_macro_diag(int x) {
  MACRODIAG();
  return x;
}

extern const int extInt = 2;
namespace intns {
extern const int extInt = 3;
}
struct S {
  int a;
};
extern const S extS = {.a = 4};
struct A {
  static const int a;
};
const int A::a = 3;
struct SC {
  const int a;
};
SC extSC = {.a = 8};
struct ST {
  static struct SC sc;
};
struct SC ST::sc = {.a = 2};
struct SCNest {
  struct SCN {
    const int a;
  } scn;
};
SCNest extSCN = {.scn = {.a = 9}};
SCNest::SCN extSubSCN = {.a = 1};
struct SCC {
  SCC(int c) : a(c) {}
  const int a;
};
SCC extSCC{7};
union U {
  const int a;
  const unsigned int b;
};
U extU = {.a = 4};

class TestAnonUnionUSR {
public:
  inline float f(int value) {
    union {
      float f;
      int i;
    };
    i = value;
    return f;
  }
  static const int Test;
};
const int TestAnonUnionUSR::Test = 5;

struct DefaultParmContext {
  static const int I;
  int f();
};

int fDefaultParm(int I = DefaultParmContext::I) {
  return I;
}

int testImportOfIncompleteDefaultParmDuringImport(int I) {
  return fDefaultParm(I);
}

const int DefaultParmContext::I = 0;

int DefaultParmContext::f() {
  return fDefaultParm();
}

class TestDelegateConstructor {
public:
  TestDelegateConstructor() : TestDelegateConstructor(2) {}
  TestDelegateConstructor(int) {}
};

int testImportOfDelegateConstructor(int i) {
  TestDelegateConstructor TDC;
  return i;
}
