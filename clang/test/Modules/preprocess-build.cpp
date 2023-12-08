// RUN: %clang_cc1 -std=c++1z -fmodules %s -verify

#pragma clang module build baz
  module baz {}
#pragma clang module endbuild // baz

#pragma clang module build foo
  module foo { module bar {} }
#pragma clang module contents
  #pragma clang module begin foo.bar
  
    // Can import baz here even though it was created in an outer build.
    #pragma clang module import baz
  
    #pragma clang module build bar
      module bar {}
    #pragma clang module contents
      #pragma clang module begin bar
        extern int n;
      #pragma clang module end
    #pragma clang module endbuild // bar
    
    #pragma clang module import bar
    
    constexpr int *f() { return &n; }
  
  #pragma clang module end
#pragma clang module endbuild // foo

#pragma clang module import bar
#pragma clang module import foo.bar
static_assert(f() == &n);

#pragma clang module build // expected-error {{expected module name}}
#pragma clang module build unterminated // expected-error {{no matching '#pragma clang module endbuild'}}
