// RUN: %clang_cc1 -fsycl-is-device -verify=expected,def -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-opencl-unknown -verify -fsyntax-only %s

void bar(int &Data) {}
void bar2(int &Data) {}
void bar(int [[clang::sycl_private]] &Data) {}
void foo(int *Data) {}
void foo2(int *Data) {}
void foo(int [[clang::sycl_private]] *Data) {}
void baz(int [[clang::sycl_private]] *Data) {} // expected-note {{candidate function not viable: cannot pass pointer to generic address space as a pointer to address space '[[clang::sycl_private]]' in 1st argument}}

template <typename T>
void tmpl(T *t) {}

void usages() {
  int [[clang::sycl_global]] *GLOB;
  int [[clang::sycl_private]] *PRIV;
  int [[clang::sycl_local]] *LOC;
  int [[clang::sycl_constant]] *CONST;
  int *NoAS;

  GLOB = PRIV;                                                     // expected-error {{assigning '[[clang::sycl_private]] int *' to '[[clang::sycl_global]] int *' changes address space of pointer}}
  GLOB = LOC;                                                      // expected-error {{assigning '[[clang::sycl_local]] int *' to '[[clang::sycl_global]] int *' changes address space of pointer}}
  PRIV = static_cast<int [[clang::sycl_private]] *>(GLOB); // expected-error {{static_cast from '[[clang::sycl_global]] int *' to '[[clang::sycl_private]] int *' is not allowed}}
  PRIV = static_cast<int [[clang::sycl_private]] *>(LOC);  // expected-error {{static_cast from '[[clang::sycl_local]] int *' to '[[clang::sycl_private]] int *' is not allowed}}
  NoAS = GLOB + PRIV;                                              // expected-error {{invalid operands to binary expression ('[[clang::sycl_global]] int *' and '[[clang::sycl_private]] int *')}}
  NoAS = GLOB + LOC;                                               // expected-error {{invalid operands to binary expression ('[[clang::sycl_global]] int *' and '[[clang::sycl_local]] int *')}}
  NoAS += GLOB;                                                    // expected-error {{invalid operands to binary expression ('int *' and '[[clang::sycl_global]] int *')}}

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
  int [[clang::sycl_local]] *l = NoAS; // expected-error {{cannot initialize a variable of type '[[clang::sycl_local]] int *' with an lvalue of type 'int *'}}

  // Explicit casts between disjoint address spaces are disallowed
  GLOB = (int [[clang::sycl_global]] *)PRIV; // expected-error {{C-style cast from '[[clang::sycl_private]] int *' to '[[clang::sycl_global]] int *' converts between mismatching address spaces}}

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
  GLOB_HOST = GLOB; // expected-error {{assigning '[[clang::sycl_global]] int *' to '__global_host int *' changes address space of pointer}}
  GLOB_HOST = static_cast<__attribute__((opencl_global_host)) int *>(GLOB); // expected-error {{static_cast from '[[clang::sycl_global]] int *' to '__global_host int *' is not allowed}}
  __attribute__((opencl_global_device)) int *GLOB_DEVICE;
  bar(*GLOB_DEVICE);
  bar2(*GLOB_DEVICE);
  GLOB = GLOB_DEVICE;
  GLOB_DEVICE = GLOB; // expected-error {{assigning '[[clang::sycl_global]] int *' to '__global_device int *' changes address space of pointer}}
  GLOB_DEVICE = static_cast<__attribute__((opencl_global_device)) int *>(GLOB); // expected-error {{static_cast from '[[clang::sycl_global]] int *' to '__global_device int *' is not allowed}}

  // Test sycl_constant conversions
  // constant -> constant: OK
  int [[clang::sycl_constant]] *c2 = CONST;
  (void)c2;

  GLOB = CONST; // expected-error {{assigning '[[clang::sycl_constant]] int *' to '[[clang::sycl_global]] int *' changes address space of pointer}}
  PRIV = CONST; // expected-error {{assigning '[[clang::sycl_constant]] int *' to '[[clang::sycl_private]] int *' changes address space of pointer}}
  LOC = CONST;  // expected-error {{assigning '[[clang::sycl_constant]] int *' to '[[clang::sycl_local]] int *' changes address space of pointer}}
  NoAS = CONST; // expected-error {{assigning '[[clang::sycl_constant]] int *' to 'int *' changes address space of pointer}}
  CONST = NoAS; // expected-error {{assigning 'int *' to '[[clang::sycl_constant]] int *' changes address space of pointer}}
  CONST = GLOB; // expected-error {{assigning '[[clang::sycl_global]] int *' to '[[clang::sycl_constant]] int *' changes address space of pointer}}
  CONST = PRIV; // expected-error {{assigning '[[clang::sycl_private]] int *' to '[[clang::sycl_constant]] int *' changes address space of pointer}}
  CONST = LOC;  // expected-error {{assigning '[[clang::sycl_local]] int *' to '[[clang::sycl_constant]] int *' changes address space of pointer}}

  GLOB = (int [[clang::sycl_global]] *)CONST;   // expected-error {{C-style cast from '[[clang::sycl_constant]] int *' to '[[clang::sycl_global]] int *' converts between mismatching address spaces}}
  CONST = (int [[clang::sycl_constant]] *)GLOB; // expected-error {{C-style cast from '[[clang::sycl_global]] int *' to '[[clang::sycl_constant]] int *' converts between mismatching address spaces}}
  PRIV = static_cast<int [[clang::sycl_private]] *>(CONST); // expected-error {{static_cast from '[[clang::sycl_constant]] int *' to '[[clang::sycl_private]] int *' is not allowed}}
  CONST = static_cast<int [[clang::sycl_constant]] *>(PRIV); // expected-error {{static_cast from '[[clang::sycl_private]] int *' to '[[clang::sycl_constant]] int *' is not allowed}}
}

// When targeting the OpenCL execution environment SYCL and OpenCL address space
// attributes are aligned. Corresponding address are mutually convertible.
void opencl_conv() {
  int [[clang::sycl_global]] *SGLOB;
  int [[clang::sycl_local]] *SLOC;
  int [[clang::sycl_private]] *SPRIV;
  int [[clang::sycl_constant]] *SCONST;
  int [[clang::sycl_generic]] *SGEN;

  __attribute__((opencl_global)) int *OGLOB;
  __attribute__((opencl_local)) int *OLOC;
  __attribute__((opencl_private)) int *OPRIV;
  __attribute__((opencl_constant)) int *OCONST;
  __attribute__((opencl_generic)) int *OGEN;

  // Corresponding SYCL/OpenCL address spaces are mutually convertible when
  // targeting OpenCL. An error is generated when not targeting OpenCL.
  OGLOB = SGLOB;   // def-error {{assigning '[[clang::sycl_global]] int *' to '__global int *' changes address space of pointer}}
  SGLOB = OGLOB;   // def-error {{assigning '__global int *' to '[[clang::sycl_global]] int *' changes address space of pointer}}
  OLOC = SLOC;     // def-error {{assigning '[[clang::sycl_local]] int *' to '__local int *' changes address space of pointer}}
  SLOC = OLOC;     // def-error {{assigning '__local int *' to '[[clang::sycl_local]] int *' changes address space of pointer}}
  OPRIV = SPRIV;   // def-error {{assigning '[[clang::sycl_private]] int *' to '__private int *' changes address space of pointer}}
  SPRIV = OPRIV;   // def-error {{assigning '__private int *' to '[[clang::sycl_private]] int *' changes address space of pointer}}
  OGEN = SGEN;     // def-error {{assigning '[[clang::sycl_generic]] int *' to '__generic int *' changes address space of pointer}}
  SGEN = OGEN;     // def-error {{assigning '__generic int *' to '[[clang::sycl_generic]] int *' changes address space of pointer}}
  OCONST = SCONST; // def-error {{assigning '[[clang::sycl_constant]] int *' to '__constant int *' changes address space of pointer}}
  SCONST = OCONST; // def-error {{assigning '__constant int *' to '[[clang::sycl_constant]] int *' changes address space of pointer}}

  // Non-corresponding SYCL/OpenCL address spaces remain disjoint and are
  // diagnosed regardless of the target.
  OGLOB = SLOC;   // expected-error {{assigning '[[clang::sycl_local]] int *' to '__global int *' changes address space of pointer}}
  SGLOB = OPRIV;  // expected-error {{assigning '__private int *' to '[[clang::sycl_global]] int *' changes address space of pointer}}
  OCONST = SGLOB; // expected-error {{assigning '[[clang::sycl_global]] int *' to '__constant int *' changes address space of pointer}}
}
