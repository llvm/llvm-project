// RUN: %clang_cc1 -fsyntax-only -fdelayed-typo-correction -verify %s

namespace c { double xxx; }
namespace d { float xxx; }
namespace z { namespace xxx {} }

void crash() {
  switch (xxx) {} // expected-error{{use of undeclared identifier 'xxx'}}
}
