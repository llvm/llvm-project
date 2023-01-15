// RUN: %clang_cc1 -fsyntax-only -verify -Wshorten-64-to-32 -triple x86_64-apple-darwin %s

int test0(long v) {
  return v; // expected-warning {{implicit conversion loses integer precision}}
}


// rdar://9546171
typedef int  int4  __attribute__ ((vector_size(16)));
typedef long long long2 __attribute__((__vector_size__(16)));

int4 test1(long2 a) {
  int4  v127 = a;  // no warning.
  return v127; 
}

int test2(long v) {
  return v / 2; // expected-warning {{implicit conversion loses integer precision: 'long' to 'int'}}
}

// rdar://10466193
void test3(int i, long long ll) {
  i += ll; // expected-warning {{implicit conversion loses integer precision}}
  i -= ll; // expected-warning {{implicit conversion loses integer precision}}
  i *= ll; // expected-warning {{implicit conversion loses integer precision}}
  i /= ll; // expected-warning {{implicit conversion loses integer precision}}
}

void test4(int i, long long ll) {
  i += i-ll; // expected-warning {{implicit conversion loses integer precision}}
  i += i+ll; // expected-warning {{implicit conversion loses integer precision}}
  i -= i-ll; // expected-warning {{implicit conversion loses integer precision}}
  i -= i+ll; // expected-warning {{implicit conversion loses integer precision}}
}

void test5(int i, int j, long long ll) {
  i += (i-j)*ll; // expected-warning {{implicit conversion loses integer precision}}
  i += (i+j)*ll; // expected-warning {{implicit conversion loses integer precision}}
  i -= ll/(i-j); // expected-warning {{implicit conversion loses integer precision}}
  i -= ll/(i-j); // expected-warning {{implicit conversion loses integer precision}}
}

void test6(char c) {
  c <<= 999999; // expected-warning {{shift count >= width of type}} \
                  // cxx17-warning {{shift count >= width of type}} \
                  // ref-warning {{shift count >= width of type}} \
                  // ref-cxx17-warning {{shift count >= width of type}}
  c >>= 999999; // expected-warning {{shift count >= width of type}} \
                  // cxx17-warning {{shift count >= width of type}} \
                  // ref-warning {{shift count >= width of type}} \
                  // ref-cxx17-warning {{shift count >= width of type}}
}
