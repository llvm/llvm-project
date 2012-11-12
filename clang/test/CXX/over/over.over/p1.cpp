// RUN: %clang_cc1 -fsyntax-only %s

template<typename T> T f0(T);
int f0(int);

// -- an object or reference being initialized 
struct S {
  int (*f0)(int);
  float (*f1)(float);
};

void test_init_f0() {
  int (*f0a)(int) = f0;
  int (*f0b)(int) = &f0;
  int (*f0c)(int) = (f0);
  float (*f0d)(float) = f0;
  float (*f0e)(float) = &f0;
  float (*f0f)(float) = (f0);
  int (&f0g)(int) = f0;
  int (&f0h)(int) = (f0);
  float (&f0i)(float) = f0;
  float (&f0j)(float) = (f0);
  S s = { f0, f0 };
}

// -- the left side of an assignment (5.17),
void test_assign_f0() {
  int (*f0a)(int) = 0;
  float (*f0b)(float) = 0;
  
  f0a = f0;
  f0a = &f0;
  f0a = (f0);
  f0b = f0;
  f0b = &f0;
  f0b = (f0);  
}

// -- a parameter of a function (5.2.2),
void eat_f0(int a(int), float (*b)(float), int (&c)(int), float (&d)(float));

void test_pass_f0() {
  eat_f0(f0, f0, f0, f0);
  eat_f0(&f0, &f0, (f0), (f0));
}

// -- a parameter of a user-defined operator (13.5),
struct X { };
void operator+(X, int(int));
void operator-(X, float(*)(float));
void operator*(X, int (&)(int));
void operator/(X, float (&)(float));

void test_operator_pass_f0(X x) {
  x + f0;
  x + &f0;
  x - f0;
  x - &f0;
  x * f0;
  x * (f0);
  x / f0;
  x / (f0);
}

// -- the return value of a function, operator function, or conversion (6.6.3),
int (*test_return_f0_a())(int) { return f0; }
int (*test_return_f0_b())(int) { return &f0; }
int (*test_return_f0_c())(int) { return (f0); }
float (*test_return_f0_d())(float) { return f0; }
float (*test_return_f0_e())(float) { return &f0; }
float (*test_return_f0_f())(float) { return (f0); }

// -- an explicit type conversion (5.2.3, 5.2.9, 5.4), or
void test_convert_f0() {
  (void)((int (*)(int))f0);
  (void)((int (*)(int))&f0);
  (void)((int (*)(int))(f0));
  (void)((float (*)(float))f0);
  (void)((float (*)(float))&f0);
  (void)((float (*)(float))(f0));
}

// -- a non-type template-parameter(14.3.2).
template<int(int)> struct Y0 { };
template<float(float)> struct Y1 { };
template<int (&)(int)> struct Y2 { };
template<float (&)(float)> struct Y3 { };

Y0<f0> y0;
Y0<&f0> y0a;
Y1<f0> y1;
Y1<&f0> y1a;
Y2<f0> y2;
Y3<f0> y3;
