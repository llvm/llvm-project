// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c++ -std=c++14 -fms-extensions -Wno-pragma-pack -fexceptions -fcxx-exceptions %s

// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -x c++ -std=c++14 -fms-extensions -Wno-pragma-pack -fexceptions -fcxx-exceptions %s

// expected-error@+1 {{expected an OpenMP directive}}
#pragma omp declare

int foo();

template <typename T>
T foofoo(); // expected-note 2 {{declared here}}

#pragma omp declare variant                                  // expected-error {{expected '(' after 'declare variant'}}
#pragma omp declare variant(                                 // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo                              // expected-error {{expected ')'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}} expected-note {{to match this '('}}
#pragma omp declare variant(x)                               // expected-error {{use of undeclared identifier 'x'}}
#pragma omp declare variant(foo)                             // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>)                    // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) xxx                // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match              // expected-error {{expected '(' after 'match'}}
#pragma omp declare variant(foofoo <int>) match(             // expected-error {{expected context selector in 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match()            // expected-error {{expected context selector in 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match(xxx)         // expected-error {{expected '=' after 'xxx' context selector set name on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match(xxx =)       // expected-error {{expected '{' after '='}}
#pragma omp declare variant(foofoo <int>) match(xxx = yyy)   // expected-error {{expected '{' after '='}}
#pragma omp declare variant(foofoo <int>) match(xxx = yyy }) // expected-error {{expected '{' after '='}}
#pragma omp declare variant(foofoo <int>) match(xxx = {)     // expected-error {{expected '}' or ',' after ')'}} expected-error {{expected '}'}} expected-note {{to match this '{'}}
#pragma omp declare variant(foofoo <int>) match(xxx = {})
#pragma omp declare variant(foofoo <int>) match(xxx = {vvv, vvv})
#pragma omp declare variant(foofoo <int>) match(xxx = {vvv} xxx) // expected-error {{expected ','}} expected-error {{expected '=' after 'xxx' context selector set name on 'omp declare variant' directive}} expected-error {{context selector set 'xxx' is used already in the same 'omp declare variant' directive}} expected-note {{previously context selector set 'xxx' used here}}
#pragma omp declare variant(foofoo <int>) match(xxx = {vvv}) xxx // expected-warning {{extra tokens at the end of '#pragma omp declare variant' are ignored}}
#pragma omp declare variant(foofoo <int>) match(implementation={xxx}) // expected-warning {{unknown context selector in 'implementation' context selector set of 'omp declare variant' directive, ignored}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor}) // expected-error {{expected '(' after 'vendor'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-error {{expected ')' or ',' after 'vendor name'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(}) // expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-error {{expected ')' or ',' after 'vendor name'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor()}) // expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score ibm)}) // expected-error {{expected '(' after 'score'}} expected-warning {{missing ':' after context selector score clause - ignoring}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score( ibm)}) // expected-error {{expected ')' or ',' after 'vendor name'}} expected-error {{expected ')'}} expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-warning {{missing ':' after context selector score clause - ignoring}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(2 ibm)}) // expected-error {{expected ')' or ',' after 'vendor name'}} expected-error 2 {{expected ')'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-warning {{missing ':' after context selector score clause - ignoring}} expected-note 2 {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(foofoo <int>()) ibm)}) // expected-warning {{missing ':' after context selector score clause - ignoring}} expected-error {{expression is not an integral constant expression}} expected-note {{non-constexpr function 'foofoo<int>' cannot be used in a constant expression}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(5): ibm), vendor(llvm)}) // expected-error {{context trait selector 'vendor' is used already in the same 'implementation' context selector set of 'omp declare variant' directive}} expected-note {{previously context trait selector 'vendor' used here}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(5): ibm), kind(cpu)}) // expected-warning {{unknown context selector in 'implementation' context selector set of 'omp declare variant' directive, ignored}}
#pragma omp declare variant(foofoo <int>) match(device={xxx}) // expected-warning {{unknown context selector in 'device' context selector set of 'omp declare variant' directive, ignored}}
#pragma omp declare variant(foofoo <int>) match(device={kind}) // expected-error {{expected '(' after 'kind'}} expected-error {{expected 'host', 'nohost', 'cpu', 'gpu', or 'fpga' in 'kind' context selector of 'device' selector set of 'omp declare variant' directive}} expected-error {{expected ')'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind(}) // expected-error {{expected 'host', 'nohost', 'cpu', 'gpu', or 'fpga' in 'kind' context selector of 'device' selector set of 'omp declare variant' directive}} expected-error 2 {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind()}) // expected-error {{expected 'host', 'nohost', 'cpu', 'gpu', or 'fpga' in 'kind' context selector of 'device' selector set of 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score cpu)}) // expected-error {{expected ')' or ',' after 'score'}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score( ibm)}) // expected-error 2 {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(2 gpu)}) // expected-error 2 {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(foofoo <int>()) ibm)}) // expected-error {{expected ')' or ',' after 'score'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(5): host), kind(llvm)}) // expected-error {{context trait selector 'kind' is used already in the same 'device' context selector set of 'omp declare variant' directive}} expected-note {{previously context trait selector 'kind' used here}} expected-error {{expected ')' or ',' after 'score'}} expected-note {{to match this '('}} expected-error {{expected ')'}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}} expected-error {{unknown 'llvm' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(5): nohost), vendor(llvm)}) // expected-warning {{unknown context selector in 'device' context selector set of 'omp declare variant' directive, ignored}} expected-error {{expected ')' or ',' after 'score'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
int bar();

#pragma omp declare variant                            // expected-error {{expected '(' after 'declare variant'}}
#pragma omp declare variant(                           // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <T>                 // expected-error {{expected ')'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}} expected-note {{to match this '('}}
#pragma omp declare variant(x)                         // expected-error {{use of undeclared identifier 'x'}}
#pragma omp declare variant(foo)                       // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo)                    // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>)                // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) xxx            // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) match          // expected-error {{expected '(' after 'match'}}
#pragma omp declare variant(foofoo <T>) match(         // expected-error {{expected context selector in 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) match()        // expected-error {{expected context selector in 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) match(xxx)     // expected-error {{expected '=' after 'xxx' context selector set name on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) match(xxx =)   // expected-error {{expected '{' after '='}}
#pragma omp declare variant(foofoo <T>) match(xxx = {) // expected-error {{expected '}' or ',' after ')'}} expected-error {{expected '}'}} expected-note {{to match this '{'}}
#pragma omp declare variant(foofoo <T>) match(xxx = {})
#pragma omp declare variant(foofoo <T>) match(xxx = {vvv, vvv})
#pragma omp declare variant(foofoo <T>) match(user = {score(<expr>) : condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(user = {score(<expr>) : condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(user = {condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(user = {condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(xxx = {vvv} xxx) // expected-error {{expected ','}} expected-error {{expected '=' after 'xxx' context selector set name on 'omp declare variant' directive}} expected-error {{context selector set 'xxx' is used already in the same 'omp declare variant' directive}} expected-note {{previously context selector set 'xxx' used here}}
#pragma omp declare variant(foofoo <T>) match(xxx = {vvv}) xxx // expected-warning {{extra tokens at the end of '#pragma omp declare variant' are ignored}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score ibm)}) // expected-error {{expected '(' after 'score'}} expected-warning {{missing ':' after context selector score clause - ignoring}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score( ibm)}) // expected-error {{expected ')' or ',' after 'vendor name'}} expected-error {{expected ')'}} expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-warning {{missing ':' after context selector score clause - ignoring}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(C ibm)}) // expected-error {{expected ')' or ',' after 'vendor name'}} expected-error 2 {{expected ')'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-warning {{missing ':' after context selector score clause - ignoring}} expected-note 2 {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(foofoo <int>()) ibm)}) // expected-warning {{missing ':' after context selector score clause - ignoring}} expected-error {{expression is not an integral constant expression}} expected-note {{non-constexpr function 'foofoo<int>' cannot be used in a constant expression}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(C+5): ibm), vendor(llvm)}) // expected-error {{context trait selector 'vendor' is used already in the same 'implementation' context selector set of 'omp declare variant' directive}} expected-note {{previously context trait selector 'vendor' used here}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(5): ibm), kind(cpu)}) // expected-warning {{unknown context selector in 'implementation' context selector set of 'omp declare variant' directive, ignored}}
#pragma omp declare variant(foofoo <int>) match(device={xxx}) // expected-warning {{unknown context selector in 'device' context selector set of 'omp declare variant' directive, ignored}}
#pragma omp declare variant(foofoo <int>) match(device={kind}) // expected-error {{expected '(' after 'kind'}} expected-error {{expected 'host', 'nohost', 'cpu', 'gpu', or 'fpga' in 'kind' context selector of 'device' selector set of 'omp declare variant' directive}} expected-error {{expected ')'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind(}) // expected-error {{expected 'host', 'nohost', 'cpu', 'gpu', or 'fpga' in 'kind' context selector of 'device' selector set of 'omp declare variant' directive}} expected-error 2 {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind()}) // expected-error {{expected 'host', 'nohost', 'cpu', 'gpu', or 'fpga' in 'kind' context selector of 'device' selector set of 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score cpu)}) // expected-error {{expected ')' or ',' after 'score'}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score( ibm)}) // expected-error 2 {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(C gpu)}) // expected-error 2 {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(foofoo <int>()) ibm)}) // expected-error {{expected ')' or ',' after 'score'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(C+5): host), kind(llvm)}) // expected-error {{context trait selector 'kind' is used already in the same 'device' context selector set of 'omp declare variant' directive}} expected-note {{previously context trait selector 'kind' used here}} expected-error {{expected ')' or ',' after 'score'}} expected-note {{to match this '('}} expected-error {{expected ')'}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}} expected-error {{unknown 'llvm' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(C+5): nohost), vendor(llvm)}) // expected-warning {{unknown context selector in 'device' context selector set of 'omp declare variant' directive, ignored}} expected-error {{expected ')' or ',' after 'score'}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{unknown 'score' device kind trait in the 'device' context selector set, expected one of 'host', 'nohost', 'cpu', 'gpu' or 'fpga'}}
template <typename T, int C>
T barbar();

// expected-error@+2 {{'#pragma omp declare variant' can only be applied to functions}}
#pragma omp declare variant(barbar <int>) match(xxx = {})
int a;
// expected-error@+2 {{'#pragma omp declare variant' can only be applied to functions}}
#pragma omp declare variant(barbar <int>) match(xxx = {})
#pragma omp threadprivate(a)
int var;
#pragma omp threadprivate(var)

// expected-error@+2 {{expected an OpenMP directive}} expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(barbar <int>) match(xxx = {})
#pragma omp declare

// expected-error@+3 {{function declaration is expected after 'declare variant' directive}}
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(barbar <int>) match(xxx = {})
#pragma omp declare variant(barbar <int>) match(xxx = {})
#pragma options align = packed
int main();

// expected-error@+3 {{function declaration is expected after 'declare variant' directive}}
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(barbar <int>) match(xxx = {})
#pragma omp declare variant(barbar <int>) match(xxx = {})
#pragma init_seg(compiler)
int main();

// expected-error@+1 {{single declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(barbar <int>) match(xxx = {})
int b, c;

// expected-error@+1 {{'C' does not refer to a value}}
#pragma omp declare variant(C) match(xxx = {})
// expected-note@+1 {{declared here}}
template <class C>
void h(C *hp, C *hp2, C *hq, C *lin) {
  b = 0;
}

// expected-error@+1 {{variant in '#pragma omp declare variant' with type '<overloaded function type>' is incompatible with type 'void (*)(int *, int *, int *, int *)'}}
#pragma omp declare variant(barbar <int>) match(xxx = {})
template <>
void h(int *hp, int *hp2, int *hq, int *lin);

int after_use_variant(void);
int after_use();
int bar() {
  return after_use();
}

// expected-warning@+1 {{'#pragma omp declare variant' cannot be applied for function after first usage; the original function might be used}}
#pragma omp declare variant(after_use_variant) match(xxx = {})
int after_use(void);

int fn();
int fn(int);
#pragma omp declare variant(fn) match(xxx = {})
int overload(void);

int fn1();
int fn1(int);
// expected-error@+1 {{variant in '#pragma omp declare variant' with type '<overloaded function type>' is incompatible with type 'int (*)(float)'}}
#pragma omp declare variant(fn1) match(xxx = {})
int overload1(float);

int fn_constexpr_variant();
// expected-error@+2 {{'#pragma omp declare variant' does not support constexpr functions}}
#pragma omp declare variant(fn_constexpr_variant) match(xxx = {})
constexpr int fn_constexpr();

constexpr int fn_constexpr_variant1();
// expected-error@+1 {{'#pragma omp declare variant' does not support constexpr functions}}
#pragma omp declare variant(fn_constexpr_variant1) match(xxx = {})
int fn_constexpr1();

int fn_sc_variant();
// expected-error@+1 {{function with '#pragma omp declare variant' has a different storage class}}
#pragma omp declare variant(fn_sc_variant) match(xxx = {})
static int fn_sc();

static int fn_sc_variant1();
// expected-error@+1 {{function with '#pragma omp declare variant' has a different storage class}}
#pragma omp declare variant(fn_sc_variant1) match(xxx = {})
int fn_sc1();

int fn_inline_variant();
// expected-error@+1 {{function with '#pragma omp declare variant' has a different inline specification}}
#pragma omp declare variant(fn_inline_variant) match(xxx = {})
inline int fn_inline();

inline int fn_inline_variant1();
// expected-error@+1 {{function with '#pragma omp declare variant' has a different inline specification}}
#pragma omp declare variant(fn_inline_variant1) match(xxx = {})
int fn_inline1();

auto fn_deduced_variant() { return 0; }
#pragma omp declare variant(fn_deduced_variant) match(xxx = {})
int fn_deduced();

int fn_deduced_variant1();
#pragma omp declare variant(fn_deduced_variant1) match(xxx = {})
auto fn_deduced1() { return 0; }

auto fn_deduced3() { return 0; }
// expected-warning@+1 {{'#pragma omp declare variant' cannot be applied to the function that was defined already; the original function might be used}}
#pragma omp declare variant(fn_deduced_variant1) match(xxx = {})
auto fn_deduced3();

auto fn_deduced_variant2() { return 0; }
// expected-error@+1 {{variant in '#pragma omp declare variant' with type 'int ()' is incompatible with type 'float (*)()'}}
#pragma omp declare variant(fn_deduced_variant2) match(xxx = {})
float fn_deduced2();

// expected-error@+1 {{exception specification in declaration does not match previous declaration}}
int fn_except_variant() noexcept(true);
// expected-note@+2 {{previous declaration is here}}
#pragma omp declare variant(fn_except_variant) match(xxx = {})
int fn_except() noexcept(false);

// expected-error@+1 {{exception specification in declaration does not match previous declaration}}
int fn_except_variant1() noexcept(false);
// expected-note@+2 {{previous declaration is here}}
#pragma omp declare variant(fn_except_variant1) match(xxx = {})
int fn_except1() noexcept(true);

struct SpecialFuncs {
  void vd();
  // expected-error@+2 {{'#pragma omp declare variant' does not support constructors}}
#pragma omp declare variant(SpecialFuncs::vd) match(xxx = {})
  SpecialFuncs();
  // expected-error@+2 {{'#pragma omp declare variant' does not support destructors}}
#pragma omp declare variant(SpecialFuncs::vd) match(xxx = {})
  ~SpecialFuncs();

  void baz();
  void bar();
  void bar(int);
#pragma omp declare variant(SpecialFuncs::baz) match(xxx = {})
#pragma omp declare variant(SpecialFuncs::bar) match(xxx = {})
  void foo1();
  SpecialFuncs& foo(const SpecialFuncs&);
  SpecialFuncs& bar(SpecialFuncs&&);
  // expected-error@+2 {{'#pragma omp declare variant' does not support defaulted functions}}
#pragma omp declare variant(SpecialFuncs::foo) match(xxx = {})
  SpecialFuncs& operator=(const SpecialFuncs&) = default;
  // expected-error@+2 {{'#pragma omp declare variant' does not support deleted functions}}
#pragma omp declare variant(SpecialFuncs::bar) match(xxx = {})
  SpecialFuncs& operator=(SpecialFuncs&&) = delete;
};

namespace N {
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant
} // namespace N
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant
