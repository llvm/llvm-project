// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-inlining=destructors -verify -std=c++11 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-inlining=destructors -verify -std=c++17 %s

using size_t =  __typeof(sizeof(int));

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);
void clang_analyzer_warnIfReached();
void clang_analyzer_explain(int);

int a, b, c, d;

struct InlineDtor {
  static int cnt;
  static int dtorCalled;
  ~InlineDtor() {
    switch (dtorCalled % 4) {
    case 0:
      a = cnt++;
      break;
    case 1:
      b = cnt++;
      break;
    case 2:
      c = cnt++;
      break;
    case 3:
      d = cnt++;
      break;
    }

    ++dtorCalled;
  }
};

int InlineDtor::cnt = 0;
int InlineDtor::dtorCalled = 0;

void foo() {
  InlineDtor::cnt = 0;
  InlineDtor::dtorCalled = 0;
  InlineDtor arr[4];
}

void testAutoDtor() {
  foo();

  clang_analyzer_eval(a == 0); // expected-warning {{TRUE}}
  clang_analyzer_eval(b == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(c == 2); // expected-warning {{TRUE}}
  clang_analyzer_eval(d == 3); // expected-warning {{TRUE}}
}

void testDeleteDtor() {
  InlineDtor::cnt = 10;
  InlineDtor::dtorCalled = 0;

  InlineDtor *arr = new InlineDtor[4];
  delete[] arr;

  clang_analyzer_eval(a == 10); // expected-warning {{TRUE}}
  clang_analyzer_eval(b == 11); // expected-warning {{TRUE}}
  clang_analyzer_eval(c == 12); // expected-warning {{TRUE}}
  clang_analyzer_eval(d == 13); // expected-warning {{TRUE}}
}

struct MemberDtor {
  InlineDtor arr[4];
};

void testMemberDtor() {
  InlineDtor::cnt = 5;
  InlineDtor::dtorCalled = 0;

  MemberDtor *MD = new MemberDtor{};
  delete MD;

  clang_analyzer_eval(a == 5); // expected-warning {{TRUE}}
  clang_analyzer_eval(b == 6); // expected-warning {{TRUE}}
  clang_analyzer_eval(c == 7); // expected-warning {{TRUE}}
  clang_analyzer_eval(d == 8); // expected-warning {{TRUE}}
}

struct MultipleMemberDtor
{
  InlineDtor arr[4];
  InlineDtor arr2[4];
};

void testMultipleMemberDtor() {
  InlineDtor::cnt = 30;
  InlineDtor::dtorCalled = 0;

  MultipleMemberDtor *MD = new MultipleMemberDtor{};
  delete MD;

  clang_analyzer_eval(a == 34); // expected-warning {{TRUE}}
  clang_analyzer_eval(b == 35); // expected-warning {{TRUE}}
  clang_analyzer_eval(c == 36); // expected-warning {{TRUE}}
  clang_analyzer_eval(d == 37); // expected-warning {{TRUE}}
}

int EvalOrderArr[4];

struct EvalOrder
{
  int ctor = 0;
  static int dtorCalled;
  static int ctorCalled;

  EvalOrder() { ctor = ctorCalled++; };

  ~EvalOrder() { EvalOrderArr[ctor] = dtorCalled++; }
};

int EvalOrder::ctorCalled = 0;
int EvalOrder::dtorCalled = 0;

void dtorEvaluationOrder() {
  EvalOrder::ctorCalled = 0;
  EvalOrder::dtorCalled = 0;
  
  EvalOrder* eptr = new EvalOrder[4];
  delete[] eptr;

  clang_analyzer_eval(EvalOrder::dtorCalled == 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrder::dtorCalled == EvalOrder::ctorCalled); // expected-warning {{TRUE}}

  clang_analyzer_eval(EvalOrderArr[0] == 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[1] == 2); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[2] == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[3] == 0); // expected-warning {{TRUE}}
}

struct EmptyDtor {
  ~EmptyDtor(){};
};

struct DefaultDtor {
  ~DefaultDtor() = default;
};

// This function used to fail on an assertion.
void no_crash() {
  EmptyDtor* eptr = new EmptyDtor[4];
  delete[] eptr;
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}

  DefaultDtor* dptr = new DefaultDtor[4];
  delete[] dptr;
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}

// This snippet used to crash.
namespace crash2
{
  template <class _Tp> class unique_ptr {
  typedef _Tp *pointer;
  pointer __ptr_;

public:
  unique_ptr(pointer __p) : __ptr_(__p) {}
  ~unique_ptr() { reset(); }
  pointer get() { return __ptr_;}
  void reset() {}
};

struct S;

S *makeS();
int bar(S *x, S *y);

void foo() {
  unique_ptr<S> x(makeS()), y(makeS());
  bar(x.get(), y.get());
}

void bar() {
  foo();
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}

} // namespace crash2

// This snippet used to crash.
namespace crash3
{
struct InlineDtor {
  ~InlineDtor() {}
};
struct MultipleMemberDtor
{
  InlineDtor arr[4];
  InlineDtor arr2[4];
};

void foo(){
  auto *arr = new MultipleMemberDtor[4];
  delete[] arr;
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}
} // namespace crash3

namespace crash4 {
struct a {
  a *b;
};
struct c {
  a d;
  c();
  ~c() {
    for (a e = d;; e = *e.b)
      ;
  }
};
void f() { 
  c g; 
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}

} // namespace crash4

namespace crash5 {
namespace std {
template <class _Tp> class unique_ptr {
  _Tp *__ptr_;
public:
  unique_ptr(_Tp *__p) : __ptr_(__p) {}
  ~unique_ptr() {}
};
} // namespace std

int SSL_use_certificate(int *arg) {
  std::unique_ptr<int> free_x509(arg);
  {
    if (SSL_use_certificate(arg)) {
      return 0;
    }
  }
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  return 1;
}

} // namespace crash5

void zeroLength(){
  InlineDtor::dtorCalled = 0;

  auto *arr = new InlineDtor[0];
  delete[] arr;

  auto *arr2 = new InlineDtor[2][0][2];
  delete[] arr2;

  auto *arr3 = new InlineDtor[0][2][2];
  delete[] arr3;

  auto *arr4 = new InlineDtor[2][2][0];
  delete[] arr4;

  clang_analyzer_eval(InlineDtor::dtorCalled == 0); // expected-warning {{TRUE}}
}


void evalOrderPrep() {
  EvalOrderArr[0] = 0;
  EvalOrderArr[1] = 0;
  EvalOrderArr[2] = 0;
  EvalOrderArr[3] = 0;

  EvalOrder::ctorCalled = 0;
  EvalOrder::dtorCalled = 0;
}

void multidimensionalPrep(){
  EvalOrder::ctorCalled = 0;
  EvalOrder::dtorCalled = 0;

  EvalOrder arr[2][2];
}

void multidimensional(){
  evalOrderPrep();
  multidimensionalPrep();
  
  clang_analyzer_eval(EvalOrder::dtorCalled == 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrder::dtorCalled == EvalOrder::ctorCalled); // expected-warning {{TRUE}}

  clang_analyzer_eval(EvalOrderArr[0] == 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[1] == 2); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[2] == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[3] == 0); // expected-warning {{TRUE}}
}

void multidimensionalHeap() {
  evalOrderPrep();

  auto* eptr = new EvalOrder[2][2];
  delete[] eptr;

  clang_analyzer_eval(EvalOrder::dtorCalled == 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrder::dtorCalled == EvalOrder::ctorCalled); // expected-warning {{TRUE}}

  clang_analyzer_eval(EvalOrderArr[0] == 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[1] == 2); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[2] == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[3] == 0); // expected-warning {{TRUE}}
}

struct MultiWrapper{
  EvalOrder arr[2][2];
};

void multidimensionalMember(){
  evalOrderPrep();
  
  auto* mptr = new MultiWrapper;
  delete mptr;

  clang_analyzer_eval(EvalOrder::dtorCalled == 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrder::dtorCalled == EvalOrder::ctorCalled); // expected-warning {{TRUE}}

  clang_analyzer_eval(EvalOrderArr[0] == 3); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[1] == 2); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[2] == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(EvalOrderArr[3] == 0); // expected-warning {{TRUE}}
}

void *memset(void *, int, size_t);
void clang_analyzer_dumpElementCount(InlineDtor *);

void nonConstantRegionExtent(){

  InlineDtor::dtorCalled = 0;

  int x = 3;
  memset(&x, 1, sizeof(x));

  InlineDtor *arr = new InlineDtor[x];
  clang_analyzer_dumpElementCount(arr); // expected-warning {{conj_$0}}
  delete [] arr;

  //FIXME: This should be TRUE but memset also sets this
  // region to a conjured symbol.
  clang_analyzer_eval(InlineDtor::dtorCalled == 0); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}
