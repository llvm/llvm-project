// RUN: %check_clang_tidy -std=c++11 -check-suffixes=ALL,CPP11 %s bugprone-multiple-new-in-one-expression %t -- -- -target x86_64-unknown-unknown
// RUN: %check_clang_tidy -std=c++17 -check-suffixes=ALL,CPP17 %s bugprone-multiple-new-in-one-expression %t -- -- -target x86_64-unknown-unknown

namespace std {
typedef __typeof__(sizeof(0)) size_t;
enum class align_val_t : std::size_t {};
class exception {};
class bad_alloc : public exception {};
struct nothrow_t {};
extern const nothrow_t nothrow;
} // namespace std

void *operator new(std::size_t, const std::nothrow_t &) noexcept;
void *operator new(std::size_t, std::align_val_t, const std::nothrow_t &) noexcept;
void *operator new(std::size_t, void *) noexcept;
void *operator new(std::size_t, char);

struct B;

struct A { int VarI; int *PtrI; B *PtrB; };

struct B { int VarI; };

struct G {
  G(A*, B*) {}
  int operator+=(A *) { return 3; };
};

struct H {
  int *a;
  int *b;
};

int f(int);
int f(A*);
int f(A*, B*);
int f(int, B*);
int f(G, G);
int f(B*);
int f(const H &);
void f1(void *, void *);
A *g(A *);

G operator+(const G&, const G&);

void test_function_parameter(A *XA, B *XB) {
  (void)f(new A, new B);
  try {
    (void)f(new A, new B);
  }
  catch (A) {};
  try {
    (void)f(new A, new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception; order of these allocations is undefined [
    (void)f(f(new A, new B));
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    int X = f(new A, new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    X = f(new A, new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:11: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    X = 1 + f(new A, new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;

    (void)f(g(new A), new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;

    (void)f(1 + f(new A), new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:19: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (void)f(XA = new A, new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:18: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (void)f(1 + f(new A), XB = new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:19: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
  }
  catch (std::exception) {}
}

void test_operator(G *G1) {
  (void)(f(new A) + f(new B));
  try {
    (void)(f(new A) + f(new B));
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:14: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (void)f(f(new A) + f(new B));
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    int X = f(new A) + f(new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    X = f(new A) + f(new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:11: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    X = 1 + f(new A) + 1 + f(new B);
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:15: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;

    (void)(f(g(new A)) + f(new B));
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:16: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;

    (void)(f(1 + f(new A)) + f(new B));
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:20: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (void)(f(1 + f(new A)) + f(1 + f(new B)));
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:20: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;

    (void)((new A)->VarI + (new A)->VarI);

    (void)(f(new A) + ((*G1) += new A));
     // CHECK-MESSAGES-ALL: :[[@LINE-1]]:14: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
  }
  catch (std::bad_alloc) {}
}

void test_construct() {
  (void)(G(new A, new B));
  try {
    (void)(G(new A, new B));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:14: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (void)(G(new A, nullptr) + G(nullptr, new B));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:14: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    f(G(new A, nullptr), G(new A, nullptr));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:9: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;

    (void)new G(new A, nullptr);
    // CHECK-MESSAGES-CPP11: :[[@LINE-1]]:11: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    // CHECK-MESSAGES-CPP17: :[[@LINE-2]]:11: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception [
    (void)new G(nullptr, (new A)->PtrB);
    G *Z = new G(new A, nullptr);
    // CHECK-MESSAGES-CPP11: :[[@LINE-1]]:12: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    // CHECK-MESSAGES-CPP17: :[[@LINE-2]]:12: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception [
    Z = new G(g(new A), nullptr);
    // CHECK-MESSAGES-CPP11: :[[@LINE-1]]:9: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    // CHECK-MESSAGES-CPP17: :[[@LINE-2]]:9: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception [
    G *Z1, *Z2 = new G(nullptr, (new A)->PtrB), *Z3;
    // CHECK-MESSAGES-CPP11: :[[@LINE-1]]:18: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    // CHECK-MESSAGES-CPP17: :[[@LINE-2]]:18: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception [
 }
  catch (const std::bad_alloc &) {}
}

void test_new_assign() {
  A *X, *Y;
  (X = new A)->VarI = (Y = new A)->VarI;
  try {
    (X = new A)->VarI = (Y = new A)->VarI;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (new A)->VarI = (Y = new A)->VarI;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (X = new A)->VarI = (new A)->VarI;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (new A)->VarI = (new A)->VarI;
    (new A)->PtrI = new int;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:6: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
    (X = new A)->VarI += (new A)->VarI;
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:10: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
  }
  catch (...) {}
}

void test_operator_fixed_order(unsigned int L) {
  (void)(f((f(new A) || f(0)) + f(new B[L])));
  try {
    (void)(f(new A) || f(new B));
    (void)(f(new A) && f(new B));
    (void)(f(new A) || f(new B) || f(new A));

    (void)(f(new A), f(new B));

    int Y = f(0, new B) ? f(new A) : f(new B);
    Y = f(new A) ? 1 : f(new B);
    Y = f(new A) ? f(new B) : 1;

    G g{new A, new B};
    H h{new int, new int};
    f({new int, new int});
    (void)f({new A, new B}, {nullptr, nullptr});
    (void)f({new A, new B}, {new A, nullptr});
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:14: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;

    (void)(f((f(new A) || f(0)) + f(new B[L])));
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:17: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
  }
  catch (std::bad_alloc) {}
}

void test_cast() {
  try {
    f1(static_cast<void *>(new A), new B);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:28: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
  }
  catch (std::bad_alloc &) {}
}

void test_nothrow(void *P) {
  try {
    (void)f(new(std::nothrow) A, new B);
    (void)f(new A, new(std::nothrow) B);
    (void)f(new(static_cast<std::align_val_t>(8), std::nothrow) A, new B);
    (void)f(new(P) A, new B);
    (void)f(new('a') A, new B);
    // CHECK-MESSAGES-ALL: :[[@LINE-1]]:13: warning: memory allocation may leak if an other allocation is sequenced after it and throws an exception;
  }
  catch (std::exception) {}
}
