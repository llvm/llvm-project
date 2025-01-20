// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsyntax-only -verify %s

int __attribute__((target_version("crc"))) dup(void) { return 3; }
int __attribute__((target_version("default"))) dup(void) { return 1; }
//expected-error@+2 {{redefinition of 'dup'}}
//expected-note@-2 {{previous definition is here}}
int __attribute__((target_version("default"))) dup(void) { return 2; }

int __attribute__((target_version("default"))) dup1(void) { return 1; }
//expected-error@+2 {{redefinition of 'dup1'}}
//expected-note@-2 {{previous definition is here}}
int dup1(void) { return 2; }

int __attribute__((target_version("aes"))) foo(void) { return 1; }
//expected-note@+1 {{previous definition is here}}
int __attribute__((target_version("default"))) foo(void) { return 2; }

//expected-note@+1 {{previous definition is here}}
int __attribute__((target_version("sha3 + aes "))) foo(void) { return 1; }
//expected-note@-1 {{previous definition is here}}

//expected-error@+1 {{redefinition of 'foo'}}
int __attribute__((target("dotprod"))) foo(void) { return -1; }
//expected-warning@-1 {{attribute declaration must precede definition}}

//expected-error@+1 {{redefinition of 'foo'}}
int foo(void) { return 2; }

//expected-note@+1 {{previous definition is here}}
__attribute__ ((target("bf16,sve,sve2,dotprod"))) int func(void) { return 1; }
//expected-error@+1 {{redefinition of 'func'}}
__attribute__ ((target("default"))) int func(void) { return 0; }

//expected-note@+1 {{previous declaration is here}}
void __attribute__((target_version("bti+flagm2"))) one(void) {}
//expected-error@+1 {{multiversioned function redeclarations require identical target attributes}}
void __attribute__((target_version("flagm2+bti"))) one(void) {}

void __attribute__((target_version("ssbs+sha2"))) two(void) {}
void __attribute__((target_version("ssbs+fp16fml"))) two(void) {}

//expected-error@+1 {{'main' cannot be a multiversioned function}}
int __attribute__((target_version("lse"))) main(void) { return 1; }

// It is ok for the default version to appear first.
int default_first(void) { return 1; }
int __attribute__((target_version("dit"))) default_first(void) { return 2; }
int __attribute__((target_version("mops"))) default_first(void) { return 3; }

// It is ok if the default version is between other versions.
int __attribute__((target_version("simd"))) default_middle(void) {return 0; }
int __attribute__((target_version("default"))) default_middle(void) { return 1; }
int __attribute__((target_version("aes"))) default_middle(void) { return 2; }

// It is ok for the default version to be the last one.
int __attribute__((target_version("rdm"))) default_last(void) {return 0; }
int __attribute__((target_version("lse+aes"))) default_last(void) { return 1; }
int __attribute__((target_version("default"))) default_last(void) { return 2; }

// It is also ok to forward declare the default.
int __attribute__((target_version("default"))) default_fwd_declare(void);
int __attribute__((target_version("sve"))) default_fwd_declare(void) { return 0; }
int default_fwd_declare(void) { return 1; }

//expected-warning@+1 {{unsupported '' in the 'target_version' attribute string; 'target_version' attribute ignored}}
int __attribute__((target_version(""))) unsup1(void) { return 1; }
//expected-warning@+1 {{unsupported 'crc32' in the 'target_version' attribute string; 'target_version' attribute ignored}}
void __attribute__((target_version("crc32"))) unsup2(void) {}

void __attribute__((target_version("default+fp16"))) koo(void) {}
//expected-error@-1 {{function multiversioning doesn't support feature 'default'}}
void __attribute__((target_version("default+default+default"))) loo(void) {}
//expected-error@-1 {{function multiversioning doesn't support feature 'default'}}
void __attribute__((target_version("rdm+rng+crc"))) redef(void) {}
//expected-error@+2 {{redefinition of 'redef'}}
//expected-note@-2 {{previous definition is here}}
void __attribute__((target_version("rdm+rng+crc"))) redef(void) {}

int def(void);
void __attribute__((target_version("dit"))) nodef(void);
void __attribute__((target_version("ls64"))) nodef(void);
void __attribute__((target_version("aes"))) ovl(void);
void __attribute__((target_version("default"))) ovl(void);
int bar() {
  // expected-error@+2 {{reference to overloaded function could not be resolved; did you mean to call it?}}
  // expected-note@-3 {{possible target for call}}
  ovl++;
  nodef();
  return def();
}
// expected-error@+1 {{function declaration cannot become a multiversioned function after first usage}}
int __attribute__((target_version("sha2"))) def(void) { return 1; }

int __attribute__((target_version("sve"))) prot();
// expected-error@-1 {{multiversioned function must have a prototype}}

int __attribute__((target_version("aes"))) rtype(int);
// expected-error@+1 {{multiversioned function declaration has a different return type}}
float __attribute__((target_version("rdm"))) rtype(int);

int __attribute__((target_version("sha2"))) combine(void) { return 1; }
// expected-error@+1 {{multiversioned function declaration has a different calling convention}}
int __attribute__((aarch64_vector_pcs, target_version("sha3"))) combine(void) { return 2; }

// expected-error@+1 {{multiversioned function must have a prototype}}
int __attribute__((target_version("fp+aes+rcpc"))) unspec_args() { return -1; }
// expected-error@-1 {{multiversioned function must have a prototype}}
// expected-note@+1 {{function multiversioning caused by this declaration}}
int __attribute__((target_version("default"))) unspec_args() { return 0; }
int cargs() { return unspec_args(); }

// expected-error@+1 {{multiversioned function must have a prototype}}
int unspec_args_implicit_default_first();
// expected-error@-1 {{multiversioned function must have a prototype}}
// expected-note@+1 {{function multiversioning caused by this declaration}}
int __attribute__((target_version("aes"))) unspec_args_implicit_default_first() { return -1; }
// expected-note@+1 {{function multiversioning caused by this declaration}}
int __attribute__((target_version("default"))) unspec_args_implicit_default_first() { return 0; }
