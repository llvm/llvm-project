// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

void threeClauses(void) __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration)));

void twoClauses(void) __attribute__((external_source_symbol(language="Swift", defined_in="module")));

void fourClauses(void) __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration, USR="test")));

void fiveClauses(void) __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration, generated_declaration, USR="test"))); // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}

void oneClause(void) __attribute__((external_source_symbol(generated_declaration)));

void noArguments(void)
__attribute__((external_source_symbol)); // expected-error {{'external_source_symbol' attribute takes at least 1 argument}}

void namedDeclsOnly(void) {
  int (^block)(void) = ^ (void)
    __attribute__((external_source_symbol(language="Swift"))) { // expected-warning {{'external_source_symbol' attribute only applies to named declarations}}
      return 1;
  };
}

[[clang::external_source_symbol(language="Swift", defined_in="module", generated_declaration)]] void threeClauses2(void);

[[clang::external_source_symbol(language="Swift", defined_in="module")]] void twoClauses2(void);

[[clang::external_source_symbol(language="Swift", defined_in="module", USR="test", generated_declaration, generated_declaration)]] // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}
void fiveClauses2(void);

[[clang::external_source_symbol(generated_declaration)]] void oneClause2(void);

[[clang::external_source_symbol]] // expected-error {{'external_source_symbol' attribute takes at least 1 argument}}
void noArguments2(void);
