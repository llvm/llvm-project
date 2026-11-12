// RUN: %clang_cc1 -verify %s

struct Convert {
  operator unsigned() const;
};

struct NoConvert {
  operator float() const;
};

struct H {
  static _Atomic Convert conv;
  static _Atomic NoConvert noconv;
};

void foo() {
    switch(H::conv){}
    (void)__builtin_stdc_rotate_left(H::conv, H::conv);

    // expected-error@+1{{statement requires expression of integer type ('NoConvert' invalid)}}
    switch(H::noconv){}
    // expected-error@+1{{1st argument must be a scalar unsigned integer type (was 'NoConvert')}}
    (void)__builtin_stdc_rotate_left(H::noconv, H::noconv);
}

