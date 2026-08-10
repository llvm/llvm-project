// RUN: %check_clang_tidy %s bugprone-unused-return-value %t \
// RUN: -config='{CheckOptions: \
// RUN:  {bugprone-unused-return-value.CheckedFunctions: "::*"}}' \
// RUN: --

struct S1 {
  S1(){};
  S1(S1 const &);
  S1(S1 &&);
  S1 &operator=(S1 const &);
  S1 &operator=(S1 &&);
  S1 &operator+=(S1);
  S1 &operator++();
  S1 &operator++(int);
  S1 &operator--();
  S1 &operator--(int);
};

struct S2 {
  S2(){};
  S2(S2 const &);
  S2(S2 &&);
};

S2 &operator-=(S2&, int);
S2 &operator++(S2 &);
S2 &operator++(S2 &, int);

S1 returnValue();
S1 const &returnRef();

void bar() {
  returnValue();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors

  S1 a{};
  a = returnValue();
  a.operator=(returnValue());

  a = returnRef();
  a.operator=(returnRef());

  a += returnRef();

  a++;
  ++a;
  a--;
  --a;

  S2 b{};

  b -= 1;
  b++;
  ++b;
}
