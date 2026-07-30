// RUN: %clang_cc1 -fsyntax-only -verify %s

void function(void) __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration)));

__attribute__((external_source_symbol(language="Swift", defined_in="module")))
@interface I

- (void)method __attribute__((external_source_symbol(defined_in= "module")));

@end

enum E {
  CaseA __attribute__((external_source_symbol(generated_declaration))),
  CaseB __attribute__((external_source_symbol(generated_declaration, language="Swift")))
} __attribute__((external_source_symbol(language = "Swift")));

void functionCustomUSR(void) __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration, USR="s:6module17functionCustomUSRyyF")));

void functionCustomUSR2(void) __attribute__((external_source_symbol(language="Swift", defined_in="module", USR="s:6module18functionCustomUSR2yyF", generated_declaration)));

void f2(void)
__attribute__((external_source_symbol())); // expected-error {{expected 'language', 'defined_in', 'generated_declaration', or 'USR'}}
void f3(void)
__attribute__((external_source_symbol(invalid))); // expected-error {{expected 'language', 'defined_in', 'generated_declaration', or 'USR'}}
void f4(void)
__attribute__((external_source_symbol(language))); // expected-error {{expected '=' after language}}
void f5(void)
__attribute__((external_source_symbol(language=))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}
void f6(void)
__attribute__((external_source_symbol(defined_in=20))); // expected-error {{expected string literal for source container name in 'external_source_symbol' attribute}}

void f7(void)
__attribute__((external_source_symbol(generated_declaration, generated_declaration))); // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}
void f8(void)
__attribute__((external_source_symbol(language="Swift", language="Swift"))); // expected-error {{duplicate 'language' clause in an 'external_source_symbol' attribute}}
void f9(void)
__attribute__((external_source_symbol(defined_in="module", language="Swift", defined_in="foo"))); // expected-error {{duplicate 'defined_in' clause in an 'external_source_symbol' attribute}}
__attribute__((external_source_symbol(defined_in="module", language="Swift", USR="foo", USR="bar"))); // expected-error {{duplicate 'USR' clause in an 'external_source_symbol' attribute}}
void f9_1(void);

void f10(void)
__attribute__((external_source_symbol(generated_declaration, language="Swift", defined_in="foo", generated_declaration, generated_declaration, language="Swift"))); // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}

void f11(void)
__attribute__((external_source_symbol(language="Objective-C++", defined_in="Some file with spaces")));

void f12(void)
__attribute__((external_source_symbol(language="C Sharp", defined_in="file:////Hello world with spaces. cs")));

void f13(void)
__attribute__((external_source_symbol(language=Swift))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}

void f14(void)
__attribute__((external_source_symbol(=))); // expected-error {{expected 'language', 'defined_in', 'generated_declaration', or 'USR'}}

void f15(void)
__attribute__((external_source_symbol(="Swift"))); // expected-error {{expected 'language', 'defined_in', 'generated_declaration', or 'USR'}}

void f16(void)
__attribute__((external_source_symbol("Swift", "module", generated_declaration))); // expected-error {{expected 'language', 'defined_in', 'generated_declaration', or 'USR'}}

void f17(void)
__attribute__((external_source_symbol(language="Swift", "generated_declaration"))); // expected-error {{expected 'language', 'defined_in', 'generated_declaration', or 'USR'}}

void f18(void)
__attribute__((external_source_symbol(language= =))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}

void f19(void)
__attribute__((external_source_symbol(defined_in="module" language="swift"))); // expected-error {{expected ')'}} expected-note {{to match this '('}}

void f20(void)
__attribute__((external_source_symbol(defined_in="module" language="swift" generated_declaration))); // expected-error {{expected ')'}} expected-note {{to match this '('}}

void f21(void)
__attribute__((external_source_symbol(defined_in= language="swift"))); // expected-error {{expected string literal for source container name in 'external_source_symbol' attribute}}

void f22(void)
__attribute__((external_source_symbol)); // expected-error {{'external_source_symbol' attribute takes at least 1 argument}}

void f23(void)
__attribute__((external_source_symbol(defined_in=, language="swift" generated_declaration))); // expected-error {{expected string literal for source container name in 'external_source_symbol' attribute}} expected-error{{expected ')'}} expected-note{{to match this '('}}

void f24(void)
__attribute__((external_source_symbol(language = generated_declaration))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}

void f25(void)
__attribute__((external_source_symbol(defined_in=123, defined_in="module"))); // expected-error {{expected string literal for source container name in 'external_source_symbol'}} expected-error {{duplicate 'defined_in' clause in an 'external_source_symbol' attribute}}

void f26(void)
__attribute__((external_source_symbol(language=Swift, language="Swift", error))); // expected-error {{expected string literal for language name in 'external_source_symbol'}} expected-error {{duplicate 'language' clause in an 'external_source_symbol' attribute}} expected-error {{expected 'language', 'defined_in', 'generated_declaration', or 'USR'}}

void f27(void)
__attribute__((external_source_symbol(USR=f27))); // expected-error {{expected string literal for USR in 'external_source_symbol' attribute}}

void f28(void)
__attribute__((external_source_symbol(USR="")));

void f29(void)
__attribute__((external_source_symbol(language=L"Swift"))); // expected-warning {{encoding prefix 'L' on an unevaluated string literal has no effect}}

void f30(void)
__attribute__((external_source_symbol(language="Swift", language=L"Swift"))); // expected-error {{duplicate 'language' clause in an 'external_source_symbol' attribute}} \
                                                                              // expected-warning {{encoding prefix 'L' on an unevaluated string literal has no effect}}

void f31(void)
__attribute__((external_source_symbol(USR=u"foo"))); // expected-warning {{encoding prefix 'u' on an unevaluated string literal has no effect}}

void f32(void)
__attribute__((external_source_symbol(USR="foo", USR=u"foo"))); // expected-error {{duplicate 'USR' clause in an 'external_source_symbol' attribute}} \
                                                                // expected-warning {{encoding prefix 'u' on an unevaluated string literal has no effect}}

void f33(void)
__attribute__((external_source_symbol(defined_in=U"module"))); // expected-warning {{encoding prefix 'U' on an unevaluated string literal has no effect}}

void f34(void)
__attribute__((external_source_symbol(defined_in="module", defined_in=U"module"))); // expected-error {{duplicate 'defined_in' clause in an 'external_source_symbol' attribute}} \
                                                                                    // expected-warning {{encoding prefix 'U' on an unevaluated string literal has no effect}}

#if __has_attribute(external_source_symbol) != 20230206
# error "invalid __has_attribute version"
#endif
