// RUN: %clang_cc1 -fmodules -x c++-module-map %s -fmodule-name=__usr_include -verify
// RUN: %clang_cc1 -fmodules -x c++-module-map %s -fmodule-name=__usr_include -verify -DIMPORT

module __usr_include {
  module stddef {}
  module stdlib {}
}

#pragma clang module contents

// expected-no-diagnostics

#pragma clang module begin __usr_include.stddef
  #define NULL 0
#pragma clang module end

#pragma clang module begin __usr_include.stdlib
  #ifdef IMPORT
    #pragma clang module import __usr_include.stddef
  #else
    #pragma clang module begin __usr_include.stddef
      #define NULL 0
    #pragma clang module end
  #endif

  void *f() { return NULL; } // ok, NULL is visible here
#pragma clang module end
