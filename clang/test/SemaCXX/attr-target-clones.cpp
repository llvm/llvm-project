// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14

// expected-error@+2 {{attribute 'target_clones' multiversioned functions do not yet support function templates}}
template<typename T, typename U>
int __attribute__((target_clones("sse4.2", "default"))) foo(){ return 1;}

void uses_lambda() {
  // expected-error@+1 {{attribute 'target_clones' multiversioned functions do not yet support lambdas}}
  auto x = []()__attribute__((target_clones("sse4.2", "arch=ivybridge", "default"))) {};
  x();
}

// GH149372
int __attribute__((target_clones("default,avx"))) redecl(int x) { return x; }
int redecl(int);
int __attribute__((target_clones("default,avx"))) redecl(int);

int __attribute__((target_clones("default,avx"))) redecl2(int x) { return x; }
int redecl2(int);
// expected-error@+2 {{'target_clones' attribute does not match previous declaration}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target_clones("default,sse4.2"))) redecl2(int);

namespace ns {
int __attribute__((target_clones("default,avx"))) redecl3(int x) { return x; }
int redecl3(int);
int __attribute__((target_clones("default,avx"))) redecl3(int);
}

extern "C" int __attribute__((target_clones("default,avx"))) redecl4(int x) { return x; }
extern "C" int redecl4(int);
extern "C" int __attribute__((target_clones("default,avx"))) redecl4(int);

static int __attribute__((target_clones("default,avx"))) redecl5(int x) { return x; }
static int redecl5(int);
static int __attribute__((target_clones("default,avx"))) redecl5(int);

int __attribute__((target_clones("default,avx"))) redecl6(int x) { return x; }
int redecl6(int);
int __attribute__((target_clones("default,avx"))) redecl6(int);
int __attribute__((target_clones("default,avx"))) redecl6(long x) { return x; }
int redecl6(long);
