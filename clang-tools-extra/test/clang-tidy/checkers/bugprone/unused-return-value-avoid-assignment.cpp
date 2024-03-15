// RUN: %check_clang_tidy %s bugprone-unused-return-value %t \
// RUN: -config='{CheckOptions: \
// RUN:  {bugprone-unused-return-value.CheckedFunctions: "::*"}}' \
// RUN: --

struct S {
  S(){};
  S(S const &);
  S(S &&);
  S &operator=(S const &);
  S &operator=(S &&);
  S &operator+=(S);
};

S returnValue();
S const &returnRef();

void bar() {
  returnValue();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors

  S a{};
  a = returnValue();
  a.operator=(returnValue());

  a = returnRef();
  a.operator=(returnRef());

  a += returnRef();
}
