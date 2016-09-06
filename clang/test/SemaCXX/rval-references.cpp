// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 %s

typedef int&& irr;
typedef irr& ilr_c1; // Collapses to int&
typedef int& ilr;
typedef ilr&& ilr_c2; // Collapses to int&

irr ret_irr() {
  return 0; // expected-warning {{returning reference to local temporary}}
}

struct not_int {};

int over(int&);
not_int over(int&&);

int over2(const int&);
not_int over2(int&&);

struct conv_to_not_int_rvalue {
  operator not_int &&();
};

typedef void (fun_type)();
void fun();
fun_type &&make_fun();

void f() {
  int &&virr1; // expected-error {{declaration of reference variable 'virr1' requires an initializer}}
  int &&virr2 = 0;
  int &&virr3 = virr2; // expected-error {{rvalue reference to type 'int' cannot bind to lvalue of type 'int'}}
  int i1 = 0;
  int &&virr4 = i1; // expected-error {{rvalue reference to type 'int' cannot bind to lvalue of type 'int'}}
  int &&virr5 = ret_irr();
  int &&virr6 = static_cast<int&&>(i1);
  (void)static_cast<not_int&&>(i1); // expected-error {{types are not compatible}}

  int i2 = over(i1);
  not_int ni1 = over(0);
  int i3 = over(virr2);
  not_int ni2 = over(ret_irr());

  int i4 = over2(i1);
  not_int ni3 = over2(0);

  ilr_c1 vilr1 = i1;
  ilr_c2 vilr2 = i1;

  conv_to_not_int_rvalue cnir;
  not_int &&ni4 = cnir;
  not_int &ni5 = cnir; // expected-error{{non-const lvalue reference to type 'not_int' cannot bind to a value of unrelated type 'conv_to_not_int_rvalue'}}
  not_int &&ni6 = conv_to_not_int_rvalue();

  fun_type &&fun_ref = fun; // works because functions are special
  fun_type &&fun_ref2 = make_fun(); // same
  fun_type &fun_lref = make_fun(); // also special

  try {
  } catch(int&&) { // expected-error {{cannot catch exceptions by rvalue reference}}
  }
}

int&& should_warn(int i) {
  // FIXME: The stack address return test doesn't reason about casts.
  return static_cast<int&&>(i); // xpected-warning {{returning reference to temporary}}
}
int&& should_not_warn(int&& i) { // But GCC 4.4 does
  return static_cast<int&&>(i);
}


// Test the return dance. This also tests IsReturnCopyElidable.
struct MoveOnly {
  MoveOnly();
  MoveOnly(const MoveOnly&) = delete;	// expected-note {{candidate constructor}} \
  // expected-note 3{{explicitly marked deleted here}}
  MoveOnly(MoveOnly&&);	// expected-note {{candidate constructor}}
  MoveOnly(int&&);	// expected-note {{candidate constructor}}
};

MoveOnly gmo;
MoveOnly returningNonEligible() {
  int i;
  static MoveOnly mo;
  MoveOnly &r = mo;
  if (0) // Copy from global can't be elided
    return gmo; // expected-error {{call to deleted constructor}}
  else if (0) // Copy from local static can't be elided
    return mo; // expected-error {{call to deleted constructor}}
  else if (0) // Copy from reference can't be elided
    return r; // expected-error {{call to deleted constructor}}
  else // Construction from different type can't be elided
    return i; // expected-error {{no viable conversion from returned value of type 'int' to function return type 'MoveOnly'}}
}
