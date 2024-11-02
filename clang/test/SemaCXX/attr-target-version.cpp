// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14
void __attribute__((target_version("default"))) wrong_tv(void);
//expected-warning@+1 {{unsupported 'vmull' in the 'target_version' attribute string; 'target_version' attribute ignored}}
void __attribute__((target_version("vmull"))) wrong_tv(void);

void __attribute__((target_version("dotprod"))) no_def(void);
void __attribute__((target_version("rdm+fp"))) no_def(void);
void __attribute__((target_version("rcpc3"))) no_def(void);
void __attribute__((target_version("mops"))) no_def(void);
void __attribute__((target_version("rdma"))) no_def(void);

// expected-error@+1 {{no matching function for call to 'no_def'}}
void foo(void) { no_def(); }

constexpr int __attribute__((target_version("sve2"))) diff_const(void) { return 1; }
//expected-error@+1 {{multiversioned function declaration has a different constexpr specification}}
int __attribute__((target_version("sve2-bitperm"))) diff_const(void);

int __attribute__((target_version("fp"))) diff_const1(void) { return 1; }
//expected-error@+1 {{multiversioned function declaration has a different constexpr specification}}
constexpr int __attribute__((target_version("sve2-aes"))) diff_const1(void);

static int __attribute__((target_version("sve2-sha3"))) diff_link(void) { return 1; }
//expected-error@+1 {{multiversioned function declaration has a different linkage}}
int __attribute__((target_version("dpb"))) diff_link(void);

int __attribute__((target_version("memtag"))) diff_link1(void) { return 1; }
//expected-error@+1 {{multiversioned function declaration has a different linkage}}
static int __attribute__((target_version("bti"))) diff_link1(void);

int __attribute__((target_version("flagm2"))) diff_link2(void) { return 1; }
extern int __attribute__((target_version("flagm"))) diff_link2(void);

namespace {
static int __attribute__((target_version("memtag3"))) diff_link2(void) { return 2; }
int __attribute__((target_version("sve2-bitperm"))) diff_link2(void) { return 1; }
} // namespace

inline int __attribute__((target_version("sme"))) diff_inline(void) { return 1; }
//expected-error@+1 {{multiversioned function declaration has a different inline specification}}
int __attribute__((target_version("fp16"))) diff_inline(void) { return 2; }

inline int __attribute__((target_version("sme"))) diff_inline1(void) { return 1; }
int __attribute__((target_version("default"))) diff_inline1(void) { return 2; }

int __attribute__((target_version("fcma"))) diff_type1(void) { return 1; }
//expected-error@+1 {{multiversioned function declaration has a different return type}}
double __attribute__((target_version("rcpc"))) diff_type1(void);

auto __attribute__((target_version("rcpc2"))) diff_type2(void) -> int { return 1; }
//expected-error@+1 {{multiversioned function declaration has a different return type}}
auto __attribute__((target_version("sve-bf16"))) diff_type2(void) -> long { return (long)1; }

int __attribute__((target_version("fp16fml"))) diff_type3(void) noexcept(false) { return 1; }
//expected-error@+2 {{exception specification in declaration does not match previous declaration}}
//expected-note@-2 {{previous declaration is here}}
int __attribute__((target_version("sve2-sha3"))) diff_type3(void) noexcept(true) { return 2; }

template <typename T> int __attribute__((target_version("default"))) temp(T) { return 1; }

template <typename T> int __attribute__((target_version("simd"))) temp1(T) { return 1; }
// expected-error@+1 {{attribute 'target_version' multiversioned functions do not yet support function templates}}
template <typename T> int __attribute__((target_version("sha3"))) temp1(T) { return 2; }

extern "C" {
int __attribute__((target_version("aes"))) extc(void) { return 1; }
}
//expected-error@+1 {{multiversioned function declaration has a different language linkage}}
int __attribute__((target_version("lse"))) extc(void) { return 1; }

auto __attribute__((target_version("default"))) ret1(void) { return 1; }
auto __attribute__((target_version("dpb"))) ret2(void) { return 1; }
auto __attribute__((target_version("dpb2"))) ret3(void) -> int { return 1; }

class Cls {
  __attribute__((target_version("rng"))) Cls();
  __attribute__((target_version("sve-i8mm"))) ~Cls();

  Cls &__attribute__((target_version("f32mm"))) operator=(const Cls &) = default;
  Cls &__attribute__((target_version("ssbs"))) operator=(Cls &&) = delete;

  virtual void __attribute__((target_version("default"))) vfunc();
  virtual void __attribute__((target_version("sm4"))) vfunc1();
};

__attribute__((target_version("sha3"))) void Decl();
namespace Nms {
using ::Decl;
// expected-error@+3 {{declaration conflicts with target of using declaration already in scope}}
// expected-note@-4 {{target of using declaration}}
// expected-note@-3 {{using declaration}}
__attribute__((target_version("jscvt"))) void Decl();
} // namespace Nms

class Out {
  int __attribute__((target_version("bti"))) func(void);
  int __attribute__((target_version("ssbs2"))) func(void);
};
int __attribute__((target_version("bti"))) Out::func(void) { return 1; }
int __attribute__((target_version("ssbs2"))) Out::func(void) { return 2; }
// expected-error@+3 {{out-of-line definition of 'func' does not match any declaration in 'Out'}}
// expected-note@-3 {{member declaration nearly matches}}
// expected-note@-3 {{member declaration nearly matches}}
int __attribute__((target_version("rng"))) Out::func(void) { return 3; }
