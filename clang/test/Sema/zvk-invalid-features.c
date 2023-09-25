// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 %s -target-feature +v -fsyntax-only -verify

#include <riscv_vector.h>

void test_zvk_features() {
  // zvbb
  __riscv_vbrev(); // expected-error {{call to undeclared function '__riscv_vbrev'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vclz(); // expected-error {{call to undeclared function '__riscv_vclz'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vctz(); // expected-error {{call to undeclared function '__riscv_vctz'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vcpopv(); // expected-error {{call to undeclared function '__riscv_vcpopv'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vwsll(); // expected-error {{call to undeclared function '__riscv_vwsll'; ISO C99 and later do not support implicit function declarations}}

  // zvbc
  __riscv_vclmul(); // expected-error {{call to undeclared function '__riscv_vclmul'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vclmulh(); // expected-error {{call to undeclared function '__riscv_vclmulh'; ISO C99 and later do not support implicit function declarations}}

  // zvkb
  __riscv_vandn(); // expected-error {{call to undeclared function '__riscv_vandn'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vbrev8(); // expected-error {{call to undeclared function '__riscv_vbrev8'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vrev8(); // expected-error {{call to undeclared function '__riscv_vrev8'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vrol(); // expected-error {{call to undeclared function '__riscv_vrol'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vror(); // expected-error {{call to undeclared function '__riscv_vror'; ISO C99 and later do not support implicit function declarations}}

  // zvkg
  __riscv_vghsh(); // expected-error {{call to undeclared function '__riscv_vghsh'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vgmul(); // expected-error {{call to undeclared function '__riscv_vgmul'; ISO C99 and later do not support implicit function declarations}}

  // zvkned
  __riscv_vaesdf(); // expected-error {{call to undeclared function '__riscv_vaesdf'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vaesdm(); // expected-error {{call to undeclared function '__riscv_vaesdm'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vaesef(); // expected-error {{call to undeclared function '__riscv_vaesef'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vaesem(); // expected-error {{call to undeclared function '__riscv_vaesem'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vaeskf1(); // expected-error {{call to undeclared function '__riscv_vaeskf1'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vaeskf2(); // expected-error {{call to undeclared function '__riscv_vaeskf2'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vaesz(); // expected-error {{call to undeclared function '__riscv_vaesz'; ISO C99 and later do not support implicit function declarations}}

  // zvknha or zvknhb
  __riscv_vsha2ch(); // expected-error {{call to undeclared function '__riscv_vsha2ch'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vsha2cl(); // expected-error {{call to undeclared function '__riscv_vsha2cl'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vsha2ms(); // expected-error {{call to undeclared function '__riscv_vsha2ms'; ISO C99 and later do not support implicit function declarations}}

  //zvksed
  __riscv_vsm4k(); // expected-error {{call to undeclared function '__riscv_vsm4k'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vsm4r(); // expected-error {{call to undeclared function '__riscv_vsm4r'; ISO C99 and later do not support implicit function declarations}}

  // zvksh
  __riscv_vsm3c(); // expected-error {{call to undeclared function '__riscv_vsm3c'; ISO C99 and later do not support implicit function declarations}}
  __riscv_vsm3me(); // expected-error {{call to undeclared function '__riscv_vsm3me'; ISO C99 and later do not support implicit function declarations}}
}
