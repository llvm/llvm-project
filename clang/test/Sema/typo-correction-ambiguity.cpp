// RUN: %clang_cc1 -fsyntax-only -verify %s

// Check the following typo correction behavior in namespaces:
// - no typos are diagnosed when an expression has ambiguous (multiple) corrections
// - proper iteration through multiple potentially ambiguous corrections

namespace AmbiguousCorrection
{
  void method_Bar();
  void method_Foo();
  void method_Zoo();
};

void testAmbiguousNoSuggestions()
{
  AmbiguousCorrection::method_Ace(); // expected-error {{no member named 'method_Ace' in namespace 'AmbiguousCorrection'}}
}

namespace MultipleCorrectionsButNotAmbiguous
{
  int PrefixType_Name(int value);
  int PrefixType_MIN();
  int PrefixType_MAX();
};

int testMultipleCorrectionsButNotAmbiguous() {
  int val = MultipleCorrectionsButNotAmbiguous::PrefixType_Enum(0);  // expected-error {{no member named 'PrefixType_Enum' in namespace 'MultipleCorrectionsButNotAmbiguous'}}
  return val;
}
