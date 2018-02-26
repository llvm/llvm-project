// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -fdouble-square-bracket-attributes %s

void threeClauses() __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration)));

void twoClauses() __attribute__((external_source_symbol(language="Swift", defined_in="module")));

void fourClauses() __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration, generated_declaration))); // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}

void oneClause() __attribute__((external_source_symbol(generated_declaration)));

void noArguments()
__attribute__((external_source_symbol)); // expected-error {{'external_source_symbol' attribute takes at least 1 argument}}

void namedDeclsOnly() {
  int (^block)(void) = ^ (void)
    __attribute__((external_source_symbol(language="Swift"))) { // expected-warning {{'external_source_symbol' attribute only applies to named declarations}}
      return 1;
  };
}

void threeClauses2() [[clang::external_source_symbol(language="Swift", defined_in="module", generated_declaration)]];

void twoClauses2() [[clang::external_source_symbol(language="Swift", defined_in="module")]];

void fourClauses2()
[[clang::external_source_symbol(language="Swift", defined_in="module", generated_declaration, generated_declaration)]]; // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}

void oneClause2() [[clang::external_source_symbol(generated_declaration)]];

void noArguments2()
[[clang::external_source_symbol]]; // expected-error {{'external_source_symbol' attribute takes at least 1 argument}}
