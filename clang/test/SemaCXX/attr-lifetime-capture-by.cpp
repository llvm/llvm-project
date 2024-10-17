// RUN: %clang_cc1 -std=c++23 -verify %s

struct S {
  const int *x;
  void captureInt(const int&x [[clang::lifetime_capture_by(this)]]) { this->x = &x; }
};

///////////////////////////
// Test for valid usages.
///////////////////////////
[[clang::lifetime_capture_by(unknown)]] // expected-error {{'lifetime_capture_by' attribute only applies to parameters and implicit object parameters}}
void nonMember(
    const int &x1 [[clang::lifetime_capture_by(s, t)]],
    S &s,
    S &t,
    const int &x2 [[clang::lifetime_capture_by(12345 + 12)]], // expected-error {{'lifetime_capture_by' attribute argument 12345 + 12 is not a known function parameter. Must be a function parameter of one of 'this', 'global' or 'unknown'}}
    const int &x3 [[clang::lifetime_capture_by(abcdefgh)]],   // expected-error {{'lifetime_capture_by' attribute argument 'abcdefgh' is not a known function parameter. Must be a function parameter of one of 'this', 'global' or 'unknown'}}
    const int &x4 [[clang::lifetime_capture_by("abcdefgh")]], // expected-error {{'lifetime_capture_by' attribute argument "abcdefgh" is not a known function parameter. Must be a function parameter of one of 'this', 'global' or 'unknown'}}
    const int &x5 [[clang::lifetime_capture_by(this)]], // expected-error {{'lifetime_capture_by' argument references unavailable implicit 'this'}}
    const int &x6 [[clang::lifetime_capture_by()]], // expected-error {{'lifetime_capture_by' attribute specifies no capturing entity}}
    const int& x7 [[clang::lifetime_capture_by(u, 
                                               x7)]], // expected-error {{'lifetime_capture_by' argument references itself}}
    const S& u
  )
{
  s.captureInt(x1);
}

struct T {
  void member(
    const int &x [[clang::lifetime_capture_by(s)]], 
    S &s,
    S &t,            
    const int &y [[clang::lifetime_capture_by(s)]],
    const int &z [[clang::lifetime_capture_by(this, x, y)]],
    const int &u [[clang::lifetime_capture_by(global, x, s)]])
  {
    s.captureInt(x);
  }
};
