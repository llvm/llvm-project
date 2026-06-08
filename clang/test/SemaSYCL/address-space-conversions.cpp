// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

void bar(int &Data) {}
void bar2(int &Data) {}
void bar(__attribute__((sycl_private)) int &Data) {}
void foo(int *Data) {}
void foo2(int *Data) {}
void foo(__attribute__((sycl_private)) int *Data) {}
void baz(__attribute__((sycl_private)) int *Data) {} // expected-note {{candidate function not viable: cannot pass pointer to generic address space as a pointer to address space 'sycl_private' in 1st argument}}

template <typename T>
void tmpl(T *t) {}

void usages() {
  __attribute__((sycl_global)) int *GLOB;
  __attribute__((sycl_private)) int *PRIV;
  __attribute__((sycl_local)) int *LOC;
  __attribute__((sycl_constant)) int *ptr1; // expected-warning {{'sycl_constant' address space attribute is deprecated}}
  int *NoAS;

  GLOB = PRIV;                                                     // expected-error {{assigning 'sycl_private int *' to 'sycl_global int *' changes address space of pointer}}
  GLOB = LOC;                                                      // expected-error {{assigning 'sycl_local int *' to 'sycl_global int *' changes address space of pointer}}
  PRIV = static_cast<__attribute__((sycl_private)) int *>(GLOB); // expected-error {{static_cast from 'sycl_global int *' to 'sycl_private int *' is not allowed}}
  PRIV = static_cast<__attribute__((sycl_private)) int *>(LOC);  // expected-error {{static_cast from 'sycl_local int *' to 'sycl_private int *' is not allowed}}
  NoAS = GLOB + PRIV;                                              // expected-error {{invalid operands to binary expression ('sycl_global int *' and 'sycl_private int *')}}
  NoAS = GLOB + LOC;                                               // expected-error {{invalid operands to binary expression ('sycl_global int *' and 'sycl_local int *')}}
  NoAS += GLOB;                                                    // expected-error {{invalid operands to binary expression ('int *' and 'sycl_global int *')}}

  bar(*GLOB);
  bar2(*GLOB);

  bar(*PRIV);
  bar2(*PRIV);

  bar(*NoAS);
  bar2(*NoAS);

  bar(*LOC);
  bar2(*LOC);

  foo(GLOB);
  foo2(GLOB);
  foo(PRIV);
  foo2(PRIV);
  foo(NoAS);
  foo2(NoAS);
  foo(LOC);
  foo2(LOC);

  tmpl(GLOB);
  tmpl(PRIV);
  tmpl(NoAS);
  tmpl(LOC);

  // Implicit casts to named address space are disallowed
  baz(NoAS);                                   // expected-error {{no matching function for call to 'baz'}}
  __attribute__((sycl_local)) int *l = NoAS; // expected-error {{cannot initialize a variable of type 'sycl_local int *' with an lvalue of type 'int *'}}

  // Explicit casts between disjoint address spaces are disallowed
  GLOB = (__attribute__((sycl_global)) int *)PRIV; // expected-error {{C-style cast from 'sycl_private int *' to 'sycl_global int *' converts between mismatching address spaces}}

  (void)static_cast<int *>(GLOB);
  (void)static_cast<void *>(GLOB);
  int *i = GLOB;
  void *v = GLOB;
  (void)i;
  (void)v;

  __attribute__((opencl_global_host)) int *GLOB_HOST;
  bar(*GLOB_HOST);
  bar2(*GLOB_HOST);
  GLOB = GLOB_HOST;
  GLOB_HOST = GLOB; // expected-error {{assigning 'sycl_global int *' to '__global_host int *' changes address space of pointer}}
  GLOB_HOST = static_cast<__attribute__((opencl_global_host)) int *>(GLOB); // expected-error {{static_cast from 'sycl_global int *' to '__global_host int *' is not allowed}}
  __attribute__((opencl_global_device)) int *GLOB_DEVICE;
  bar(*GLOB_DEVICE);
  bar2(*GLOB_DEVICE);
  GLOB = GLOB_DEVICE;
  GLOB_DEVICE = GLOB; // expected-error {{assigning 'sycl_global int *' to '__global_device int *' changes address space of pointer}}
  GLOB_DEVICE = static_cast<__attribute__((opencl_global_device)) int *>(GLOB); // expected-error {{static_cast from 'sycl_global int *' to '__global_device int *' is not allowed}}
}
