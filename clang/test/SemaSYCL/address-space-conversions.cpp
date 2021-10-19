// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

void bar(int &Data) {}
void bar2(int &Data) {}
void bar(__attribute__((opencl_private)) int &Data) {}
void foo(int *Data) {}
void foo2(int *Data) {}
void foo(__attribute__((opencl_private)) int *Data) {}
void baz(__attribute__((opencl_private)) int *Data) {} // expected-note {{candidate function not viable: cannot pass pointer to generic address space as a pointer to address space '__private' in 1st argument}}

template <typename T>
void tmpl(T *t) {}

void usages() {
  __attribute__((opencl_global)) int *GLOB;
  __attribute__((opencl_private)) int *PRIV;
  __attribute__((opencl_local)) int *LOC;
  int *NoAS;

  GLOB = PRIV;                                                     // expected-error {{assigning '__private int *' to '__global int *' changes address space of pointer}}
  GLOB = LOC;                                                      // expected-error {{assigning '__local int *' to '__global int *' changes address space of pointer}}
  PRIV = static_cast<__attribute__((opencl_private)) int *>(GLOB); // expected-error {{static_cast from '__global int *' to '__private int *' is not allowed}}
  PRIV = static_cast<__attribute__((opencl_private)) int *>(LOC);  // expected-error {{static_cast from '__local int *' to '__private int *' is not allowed}}
  NoAS = GLOB + PRIV;                                              // expected-error {{invalid operands to binary expression ('__global int *' and '__private int *')}}
  NoAS = GLOB + LOC;                                               // expected-error {{invalid operands to binary expression ('__global int *' and '__local int *')}}
  NoAS += GLOB;                                                    // expected-error {{invalid operands to binary expression ('int *' and '__global int *')}}

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
  __attribute__((opencl_local)) int *l = NoAS; // expected-error {{cannot initialize a variable of type '__local int *' with an lvalue of type 'int *'}}

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
  GLOB_HOST = GLOB; // expected-error {{assigning '__global int *' to '__global_host int *' changes address space of pointer}}
  GLOB_HOST = static_cast<__attribute__((opencl_global_host)) int *>(GLOB); // expected-error {{static_cast from '__global int *' to '__global_host int *' is not allowed}}
  __attribute__((opencl_global_device)) int *GLOB_DEVICE;
  bar(*GLOB_DEVICE);
  bar2(*GLOB_DEVICE);
  GLOB = GLOB_DEVICE;
  GLOB_DEVICE = GLOB; // expected-error {{assigning '__global int *' to '__global_device int *' changes address space of pointer}}
  GLOB_DEVICE = static_cast<__attribute__((opencl_global_device)) int *>(GLOB); // expected-error {{static_cast from '__global int *' to '__global_device int *' is not allowed}}
}
