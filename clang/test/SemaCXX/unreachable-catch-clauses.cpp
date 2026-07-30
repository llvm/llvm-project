// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

class BaseEx {};
class Ex1: public BaseEx {};
typedef Ex1 Ex2;

void f();

void test()
try {}
catch (BaseEx &e) { f(); } // expected-note 2{{for type 'BaseEx &'}}
catch (Ex1 &e) { f(); } // expected-warning {{exception of type 'Ex1 &' will be caught by earlier handler}} \
                           expected-note {{for type 'Ex1 &'}}
// FIXME: It would be nicer to only issue one warning on the below line instead
// of two. We get two diagnostics because the first one is noticing that there
// is a class hierarchy inversion where the earlier base class handler will
// catch throwing the derived class and the second one is because Ex2 and Ex1
// are the same type after canonicalization.
catch (Ex2 &e) { f(); } // expected-warning 2{{exception of type 'Ex2 &' (aka 'Ex1 &') will be caught by earlier handler}}

namespace GH61177 {
void func() {
  const char arr[4] = "abc";

  // We should not issue an "exception will be caught by earlier handler"
  // diagnostic, as that is a lie.
  try {
    throw arr;
  } catch (char *p) {
  } catch (const char *p) {
  }
}
} // GH61177
