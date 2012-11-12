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
  const non_const_copy cncc1; // expected-error {{default initialization of an object of const type 'const non_const_copy' requires a user-provided default constructor}}
  non_const_copy ncc3 = cncc; // expected-error {{no matching}}
  ncc = cncc; // expected-error {{no viable overloaded}}
};

struct non_const_derived : non_const_copy {
  non_const_derived(const non_const_derived&) = default; // expected-error {{requires it to be non-const}}
  non_const_derived& operator =(non_const_derived&) = default;
};

struct bad_decls {
  bad_decls(volatile bad_decls&) = default; // expected-error {{may not be volatile}} expected-error {{must be defaulted outside the class}}
  bad_decls&& operator = (bad_decls) = default; // expected-error {{lvalue reference}} expected-error {{must return 'bad_decls &'}}
  bad_decls& operator = (volatile bad_decls&) = default; // expected-error {{may not be volatile}} expected-error {{must be defaulted outside the class}}
  bad_decls& operator = (const bad_decls&) const = default; // expected-error {{may not have 'const', 'constexpr' or 'volatile' qualifiers}}
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
// FIXME: This should error in the virtual override check.
// It doesn't because we generate the implicit specification later than
// appropriate.
struct except_spec_d_bad : except_spec_a, except_spec_b {
  ~except_spec_d_bad() = default;
};

// FIXME: This should error because the exceptions spec doesn't match.
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
