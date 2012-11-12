// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify -Wno-unused -Wunused-comparison %s

struct A {
  bool operator==(const A&);
  bool operator!=(const A&);
  A operator|=(const A&);
  operator bool();
};

void test() {
  int x, *p;
  A a, b;

  x == 7; // expected-warning {{equality comparison result unused}} \
          // expected-note {{use '=' to turn this equality comparison into an assignment}}
  x != 7; // expected-warning {{inequality comparison result unused}} \
          // expected-note {{use '|=' to turn this inequality comparison into an or-assignment}}
  7 == x; // expected-warning {{equality comparison result unused}}
  p == p; // expected-warning {{equality comparison result unused}} \
          // expected-note {{use '=' to turn this equality comparison into an assignment}} \
          // expected-warning {{self-comparison always evaluates to true}}
  a == a; // expected-warning {{equality comparison result unused}} \
          // expected-note {{use '=' to turn this equality comparison into an assignment}}
  a == b; // expected-warning {{equality comparison result unused}} \
          // expected-note {{use '=' to turn this equality comparison into an assignment}}
  a != b; // expected-warning {{inequality comparison result unused}} \
          // expected-note {{use '|=' to turn this inequality comparison into an or-assignment}}
  A() == b; // expected-warning {{equality comparison result unused}}
  if (42) x == 7; // expected-warning {{equality comparison result unused}} \
                  // expected-note {{use '=' to turn this equality comparison into an assignment}}
  else if (42) x == 7; // expected-warning {{equality comparison result unused}} \
                       // expected-note {{use '=' to turn this equality comparison into an assignment}}
  else x == 7; // expected-warning {{equality comparison result unused}} \
               // expected-note {{use '=' to turn this equality comparison into an assignment}}
  do x == 7; // expected-warning {{equality comparison result unused}} \
             // expected-note {{use '=' to turn this equality comparison into an assignment}}
  while (false);
  while (false) x == 7; // expected-warning {{equality comparison result unused}} \
                        // expected-note {{use '=' to turn this equality comparison into an assignment}}
  for (x == 7; // expected-warning {{equality comparison result unused}} \
               // expected-note {{use '=' to turn this equality comparison into an assignment}}
       x == 7; // No warning -- result is used
       x == 7) // expected-warning {{equality comparison result unused}} \
               // expected-note {{use '=' to turn this equality comparison into an assignment}}
    x == 7; // expected-warning {{equality comparison result unused}} \
            // expected-note {{use '=' to turn this equality comparison into an assignment}}
  switch (42) default: x == 7; // expected-warning {{equality comparison result unused}} \
                               // expected-note {{use '=' to turn this equality comparison into an assignment}}
  switch (42) case 42: x == 7; // expected-warning {{equality comparison result unused}} \
                               // expected-note {{use '=' to turn this equality comparison into an assignment}}
  switch (42) {
    case 1:
    case 2:
    default:
    case 3:
    case 4:
      x == 7; // expected-warning {{equality comparison result unused}} \
              // expected-note {{use '=' to turn this equality comparison into an assignment}}
  }

  (void)(x == 7);
  (void)(p == p); // expected-warning {{self-comparison always evaluates to true}}
  { bool b = x == 7; }

  { bool b = ({ x == 7; // expected-warning {{equality comparison result unused}} \
                        // expected-note {{use '=' to turn this equality comparison into an assignment}}
                x == 7; }); } // no warning on the second, its result is used!

#define EQ(x,y) (x) == (y)
  EQ(x, 5);
#undef EQ
}

namespace PR10291 {
  template<typename T>
  class X
  {
  public:

    X() : i(0) { } 

    void foo()
    {   
      throw 
        i == 0u ?
        5 : 6;
    }   

  private:
    int i;
  };

  X<int> x;
}
