// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace GH138850 {
void test() {
  int tmp = add(int, 0, 0);    // expected-error {{expected '(' for function-style cast or type construction}} \
                                  expected-note {{previous definition is here}}
  uint tmp = add(uint, 1, 1);  // expected-error {{use of undeclared identifier 'uint'; did you mean 'int'?}} \
                                  expected-error {{redefinition of 'tmp'}} \
                                  expected-error {{use of undeclared identifier 'uint'}}
  call(void, f, (int)tmp);     // expected-error {{expected '(' for function-style cast or type construction}} \
                                  expected-error {{use of undeclared identifier 'f'}}
}
}

namespace GH107840 {
struct tm {};          // expected-note {{'tm' declared here}}

auto getCache = [&] {  // expected-error {{non-local lambda expression cannot have a capture-default}}
  ::foo([=] {          // expected-error {{no member named 'foo' in the global namespace}}
    tms time;          // expected-error {{unknown type name 'tms'; did you mean 'tm'?}}
    (void)time;
  });
};
}

namespace GH59391 {
template <typename b> class c {
  c(b);
  b e;
  void f() {
    for (auto core : a::c(cores)) { // expected-error {{use of undeclared identifier 'cores'}} \
                                       expected-error {{use of undeclared identifier 'a'}}
    }
  }
};
}

namespace GH45915 {
short g_volatile_ushort;                   // expected-note {{'g_volatile_ushort' declared here}}
namespace a {
   int b = l_volatile_uwchar.a ::c ::~d<>; // expected-error {{use of undeclared identifier 'l_volatile_uwchar'}} \
                                              expected-error {{no member named 'd' in namespace 'GH45915::a'}}
}
}

namespace GH45891 {
int a = b.c < enum , > :: template ~d < > [ e; // expected-error {{use of undeclared identifier 'b'}} \
                                                  expected-error {{expected identifier or '{'}} \
                                                  expected-error {{expected ';' after top level declarator}}
}

namespace GH32903 {
void
B(
  char cat_dog_3, char cat_dog_2, char cat_dog_1, char cat_dog_0, char pigeon_dog_3, char pigeon_dog_2,
  char pigeon_dog_1, char pigeon_dog_0, short &elefant15_lion, short &elefant14_lion, short &elefant13_lion,       // expected-note 3 {{declared here}}
  short &elefant12_lion, short &elefant11_lion, short &elefant10_lion, short &elefant9_lion, short &elefant8_lion, // expected-note 5 {{declared here}}
  short &elefant7_lion, short &elefant6_lion, short &elefant5_lion, short &elefant4_lion, short &elefant3_lion,    // expected-note 2 {{declared here}}
  short &elefant2_lion, short &elefant1_lion, short &elefant0_lion, char& no_animal)
{

    A(  // FIXME: it's surprising that we don't issue a "use of undeclared identifier" diagnostic for the call itself.
        elefant_15_lion, elefant_14_lion, elefant_13_lion, elefant_12_lion, elefant_11_lion, elefant_10_lion, elefant_9_lion, // expected-error 7 {{use of undeclared identifier}}
        elefant_8_lion, elefant_7_lion, elefant_6_lion, elefant_5_lion, elefant_4_lion, elefant_3_lion, elefant_2_lion,       // expected-error 7 {{use of undeclared identifier}}
        elefant_1_lion, elefant_0_lion, no_animal, other_mammal);                                                             // expected-error 3 {{use of undeclared identifier}}
}
}
