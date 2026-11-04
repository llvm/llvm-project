// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-identifier-length.LineCountThreshold: 3}}' \
// RUN: -- -fexceptions

struct myexcept {
  int val;
};

template<typename... Ts>
void doIt(Ts...);

#define MY_MACRO(arg) doIt(arg, arg)

int g = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable name 'g' is too short, expected at least 3 characters [readability-identifier-length]

void shouldWarn(int z)
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: parameter name 'z' is too short, expected at least 3 characters [readability-identifier-length]
{
  int i = 5;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable name 'i' is too short, expected at least 3 characters [readability-identifier-length]
  ++i;

  for (int m = 0; m < 5; ++m)
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: loop variable name 'm' is too short, expected at least 2 characters [readability-identifier-length]
  {
    doIt(i);
    doIt(m);
  }

  try {
    doIt(z);
  } catch (const myexcept &x)
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: exception variable name 'x' is too short, expected at least 2 characters [readability-identifier-length]
  {
    doIt(x);
  }

  int a = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable name 'a' is too short, expected at least 3 characters [readability-identifier-length]
  ++a;
  MY_MACRO(a);

  int b = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable name 'b' is too short, expected at least 3 characters [readability-identifier-length]
  [&](){
    doIt(b);
  }();

  int c = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable name 'c' is too short, expected at least 3 characters [readability-identifier-length]
  [=](){
    doIt(c);
  }();
}

void shouldNotWarn(int m)
{
  doIt(m);

  int v = 5;
  ++v;
  doIt(v);

  for (int a = 0; a < 42; ++a)
  {
      doIt(a);
  }

  try {
    doIt();
  } catch (const myexcept &x) {
    doIt(x);
  }

  int a = 0;
  MY_MACRO(a);

  int b = 0;
  [&](){ doIt(b); }();

  int c = 0;
  [=](){ doIt(c); }();
}
