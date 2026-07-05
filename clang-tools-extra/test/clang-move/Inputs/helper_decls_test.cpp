#include "helper_decls_test.h"

namespace {
class HelperC1 {
public:
  static int I;
};

int HelperC1::I = 0;

class HelperC2 {};

class HelperC3 {
 public:
  static int I;
};

int HelperC3::I = 0;

void HelperFun1() {}

void HelperFun2() { HelperFun1(); }

const int K1 = 1;
} // namespace

static const int K2 = 2;
static void HelperFun3() { K2; }

namespace a {

static const int K3 = 3;
static const int K4 = HelperC3::I;
static const int K5 = 5;
static const int K6 = 6;

static void HelperFun4() {}
static void HelperFun6() {}

void Class1::f() { HelperFun2(); }

void Class2::f() {
  HelperFun1();
  HelperFun3();
}

void Class3::f() { HelperC1::I; }

void Class4::f() { HelperC2 c2; }

void Class5::f() {
  int Result = K1 + K2 + K3;
  HelperFun4();
}

int Class6::f() {
  int R = K4;
  return R;
}

int Class7::f() {
  int R = K6;
  return R;
}

int Class7::g() {
  HelperFun6();
  return 1;
}

static int HelperFun5() {
  int R = K5;
  return R;
}

void Fun1() { HelperFun5(); }

} // namespace a

namespace b {
namespace {
void HelperFun7();

class HelperC4;
} // namespace

void Fun3() {
  HelperFun7();
  HelperC4 *t;
}

namespace {
void HelperFun7() {}

class HelperC4 {};
} // namespace
} // namespace b
