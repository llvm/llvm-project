// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +i -fsyntax-only -verify -std=c2x %s

//expected-note@+1 {{previous definition is here}}
int __attribute__((target("arch=rv64g"))) foo(void) { return 0; }
//expected-error@+1 {{redefinition of 'foo'}}
int __attribute__((target("arch=rv64gc"))) foo(void) { return 0; }

//expected-warning@+1 {{unsupported 'notafeature' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=+notafeature"))) UnsupportFeature(void) { return 0; }

//expected-warning@+1 {{unsupported 'notafeature' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=-notafeature"))) UnsupportNegativeFeature(void) { return 0; }

//expected-warning@+1 {{unsupported 'arch=+zba,zbb' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=+zba,zbb"))) WithoutPlus(void) { return 0; }

//expected-warning@+1 {{unsupported 'arch=zba' in the 'target' attribute string; 'target' attribute ignored}}
int __attribute__((target("arch=zba"))) WithoutPlus2(void) { return 0; }
