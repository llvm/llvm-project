// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct non_copiable {
  non_copiable(const non_copiable&) = delete; // expected-note {{marked deleted here}}
  non_copiable& operator = (const non_copiable&) = delete; // expected-note {{explicitly deleted}}
  non_copiable() = default;
};

struct non_const_copy {
  non_const_copy(non_const_copy&);
  non_const_copy& operator = (non_const_copy&) &;
  non_const_copy& operator = (non_const_copy&) &&;
  non_const_copy() = default; // expected-note {{not viable}}
  int uninit_field;
};
non_const_copy::non_const_copy(non_const_copy&) = default; // expected-note {{not viable}}
non_const_copy& non_const_copy::operator = (non_const_copy&) & = default; // expected-note {{not viable}}
non_const_copy& non_const_copy::operator = (non_const_copy&) && = default; // expected-note {{not viable}}

void fn1 () {
  non_copiable nc;
  non_copiable nc2 = nc; // expected-error {{deleted constructor}}
  nc = nc; // expected-error {{deleted operator}}

  non_const_copy ncc;
  non_const_copy ncc2 = ncc;
  ncc = ncc2;
  const non_const_copy cncc{};
  const non_const_copy cncc1; // expected-error {{default initialization of an object of const type 'const non_const_copy' without a user-provided default constructor}}
  non_const_copy ncc3 = cncc; // expected-error {{no matching}}
  ncc = cncc; // expected-error {{no viable overloaded}}
};

struct no_fields { };
struct all_init {
  int a = 0;
  int b = 0;
};
struct some_init {
  int a = 0;
  int b;
  int c = 0;
};
struct some_init_mutable {
  int a = 0;
  mutable int b;
  int c = 0;
};
struct some_init_def {
  some_init_def() = default;
  int a = 0;
  int b;
  int c = 0;
};
struct some_init_ctor {
  some_init_ctor();
  int a = 0;
  int b;
  int c = 0;
};
struct sub_some_init : public some_init_def { };
struct sub_some_init_ctor : public some_init_def {
  sub_some_init_ctor();
};
struct sub_some_init_ctor2 : public some_init_ctor {
};
struct some_init_container {
  some_init_def sid;
};
struct some_init_container_ctor {
  some_init_container_ctor();
  some_init_def sid;
};
struct no_fields_container {
  no_fields nf;
};
struct param_pack_ctor {
  template <typename... T>
  param_pack_ctor(T...);
  int n;
};
struct param_pack_ctor_field {
  param_pack_ctor ndc;
};
struct multi_param_pack_ctor {
  template <typename... T, typename... U>
  multi_param_pack_ctor(T..., U..., int f = 0);
  int n;
};
struct ignored_template_ctor_and_def {
  template <class T> ignored_template_ctor_and_def(T* f = nullptr);
  ignored_template_ctor_and_def() = default;
  int field;
};
template<bool, typename = void> struct enable_if {};
template<typename T> struct enable_if<true, T> { typedef T type; };
struct multi_param_pack_and_defaulted {
  template <typename... T,
            typename enable_if<sizeof...(T) != 0>::type* = nullptr>
  multi_param_pack_and_defaulted(T...);
  multi_param_pack_and_defaulted() = default;
  int n;
};

void constobjs() {
  const no_fields nf; // ok
  const all_init ai; // ok
  const some_init si; // expected-error {{default initialization of an object of const type 'const some_init' without a user-provided default constructor}}
  const some_init_mutable sim; // ok
  const some_init_def sid; // expected-error {{default initialization of an object of const type 'const some_init_def' without a user-provided default constructor}}
  const some_init_ctor sic; // ok
  const sub_some_init ssi; // expected-error {{default initialization of an object of const type 'const sub_some_init' without a user-provided default constructor}}
  const sub_some_init_ctor ssic; // ok
  const sub_some_init_ctor2 ssic2; // ok
  const some_init_container sicon; // expected-error {{default initialization of an object of const type 'const some_init_container' without a user-provided default constructor}}
  const some_init_container_ctor siconc; // ok
  const no_fields_container nfc; // ok
  const param_pack_ctor ppc; // ok
  const param_pack_ctor_field ppcf; // ok
  const multi_param_pack_ctor mppc; // ok
  const multi_param_pack_and_defaulted mppad; // expected-error {{default initialization of an object of const type 'const multi_param_pack_and_defaulted' without a user-provided default constructor}}
  const ignored_template_ctor_and_def itcad; // expected-error {{default initialization of an object of const type 'const ignored_template_ctor_and_def' without a user-provided default constructor}}

}

struct non_const_derived : non_const_copy {
  non_const_derived(const non_const_derived&) = default; // expected-error {{requires it to be non-const}}
  non_const_derived& operator =(non_const_derived&) = default;
};

struct bad_decls {
  bad_decls(volatile bad_decls&) = default; // expected-error {{may not be volatile}}
  bad_decls&& operator = (bad_decls) = default; // expected-error {{lvalue reference}} expected-error {{must return 'bad_decls &'}}
  bad_decls& operator = (volatile bad_decls&) = default; // expected-error {{may not be volatile}}
  bad_decls& operator = (const bad_decls&) const = default; // expected-error {{may not have 'const', 'constexpr' or 'volatile' qualifiers}}
};

struct DefaultDelete {
  DefaultDelete() = default; // expected-note {{previous declaration is here}}
  DefaultDelete() = delete; // expected-error {{constructor cannot be redeclared}}

  ~DefaultDelete() = default; // expected-note {{previous declaration is here}}
  ~DefaultDelete() = delete; // expected-error {{destructor cannot be redeclared}}

  DefaultDelete &operator=(const DefaultDelete &) = default; // expected-note {{previous declaration is here}}
  DefaultDelete &operator=(const DefaultDelete &) = delete; // expected-error {{class member cannot be redeclared}}
};

struct DeleteDefault {
  DeleteDefault() = delete; // expected-note {{previous definition is here}}
  DeleteDefault() = default; // expected-error {{constructor cannot be redeclared}}

  ~DeleteDefault() = delete; // expected-note {{previous definition is here}}
  ~DeleteDefault() = default; // expected-error {{destructor cannot be redeclared}}

  DeleteDefault &operator=(const DeleteDefault &) = delete; // expected-note {{previous definition is here}}
  DeleteDefault &operator=(const DeleteDefault &) = default; // expected-error {{class member cannot be redeclared}}
};

struct A {}; struct B {};

struct except_spec_a {
  virtual ~except_spec_a() throw(A);
  except_spec_a() throw(A);
};
struct except_spec_b {
  virtual ~except_spec_b() throw(B);
  except_spec_b() throw(B);
};

struct except_spec_d_good : except_spec_a, except_spec_b {
  ~except_spec_d_good();
};
except_spec_d_good::~except_spec_d_good() = default;
struct except_spec_d_good2 : except_spec_a, except_spec_b {
  ~except_spec_d_good2() = default;
};
struct except_spec_d_bad : except_spec_a, except_spec_b {
  ~except_spec_d_bad() noexcept;
};
// FIXME: This should error because this exception spec is not
// compatible with the implicit exception spec.
except_spec_d_bad::~except_spec_d_bad() noexcept = default;

// FIXME: This should error because this exception spec is not
// compatible with the implicit exception spec.
struct except_spec_d_mismatch : except_spec_a, except_spec_b {
  except_spec_d_mismatch() throw(A) = default;
};
struct except_spec_d_match : except_spec_a, except_spec_b {
  except_spec_d_match() throw(A, B) = default;
};

// gcc-compatibility: allow attributes on default definitions
// (but not normal definitions)
struct S { S(); };
S::S() __attribute((pure)) = default;

using size_t = decltype(sizeof(0));
void *operator new(size_t) = delete; // expected-error {{deleted definition must be first declaration}} expected-note {{implicit}}
void operator delete(void *) noexcept = delete; // expected-error {{deleted definition must be first declaration}} expected-note {{implicit}}
