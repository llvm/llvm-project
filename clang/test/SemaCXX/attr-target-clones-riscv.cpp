// RUN: %clang_cc1 -triple riscv64-linux-gnu  -fsyntax-only -verify -fexceptions -fcxx-exceptions %s -std=c++14

// expected-warning@+1 {{unsupported 'mcpu=sifive-u74' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void __attribute__((target_clones("default", "mcpu=sifive-u74"))) mcpu() {}

// expected-warning@+1 {{unsupported 'mtune=sifive-u74' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void __attribute__((target_clones("default", "mtune=sifive-u74"))) mtune() {}

// expected-warning@+1 {{version list contains duplicate entries}}
void __attribute__((target_clones("default", "arch=+c", "arch=+c"))) dupVersion() {}

// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void __attribute__((target_clones("default", ""))) emptyVersion() {}

// expected-error@+1 {{'target_clones' multiversioning requires a default target}}
void __attribute__((target_clones("arch=+c"))) withoutDefault() {}

// expected-warning@+1 {{unsupported '+c' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void __attribute__((target_clones("default", "+c"))) invaildVersion() {}

// expected-warning@+1 {{unsupported 'arch=rv64g' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void __attribute__((target_clones("default", "arch=rv64g"))) fullArchString() {}

// expected-warning@+1 {{unsupported 'arch=+zicsr' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void __attribute__((target_clones("default", "arch=+zicsr"))) UnsupportBitMaskExt() {}


void lambda() {
  // expected-error@+1 {{attribute 'target_clones' multiversioned functions do not yet support lambdas}}
  auto x = []() __attribute__((target_clones("default"))){};
  x();
  // expected-error@+1 {{attribute 'target_clones' multiversioned functions do not yet support lambdas}}
  auto y = []() __attribute__((target_clones("arch=+v", "default"))){};
  y();
}
