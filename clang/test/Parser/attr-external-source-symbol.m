// RUN: %clang_cc1 -fsyntax-only -verify %s

void function() __attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration)));

__attribute__((external_source_symbol(language="Swift", defined_in="module")))
@interface I

- (void)method __attribute__((external_source_symbol(defined_in= "module")));

@end

enum E {
  CaseA __attribute__((external_source_symbol(generated_declaration))),
  CaseB __attribute__((external_source_symbol(generated_declaration, language="Swift")))
} __attribute__((external_source_symbol(language = "Swift")));

void f2()
__attribute__((external_source_symbol())); // expected-error {{expected 'language', 'defined_in', or 'generated_declaration'}}
void f3()
__attribute__((external_source_symbol(invalid))); // expected-error {{expected 'language', 'defined_in', or 'generated_declaration'}}
void f4()
__attribute__((external_source_symbol(language))); // expected-error {{expected '=' after language}}
void f5()
__attribute__((external_source_symbol(language=))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}
void f6()
__attribute__((external_source_symbol(defined_in=20))); // expected-error {{expected string literal for source container name in 'external_source_symbol' attribute}}

void f7()
__attribute__((external_source_symbol(generated_declaration, generated_declaration))); // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}
void f8()
__attribute__((external_source_symbol(language="Swift", language="Swift"))); // expected-error {{duplicate 'language' clause in an 'external_source_symbol' attribute}}
void f9()
__attribute__((external_source_symbol(defined_in="module", language="Swift", defined_in="foo"))); // expected-error {{duplicate 'defined_in' clause in an 'external_source_symbol' attribute}}

void f10()
__attribute__((external_source_symbol(generated_declaration, language="Swift", defined_in="foo", generated_declaration, generated_declaration, language="Swift"))); // expected-error {{duplicate 'generated_declaration' clause in an 'external_source_symbol' attribute}}

void f11()
__attribute__((external_source_symbol(language="Objective-C++", defined_in="Some file with spaces")));

void f12()
__attribute__((external_source_symbol(language="C Sharp", defined_in="file:////Hello world with spaces. cs")));

void f13()
__attribute__((external_source_symbol(language=Swift))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}

void f14()
__attribute__((external_source_symbol(=))); // expected-error {{expected 'language', 'defined_in', or 'generated_declaration'}}

void f15()
__attribute__((external_source_symbol(="Swift"))); // expected-error {{expected 'language', 'defined_in', or 'generated_declaration'}}

void f16()
__attribute__((external_source_symbol("Swift", "module", generated_declaration))); // expected-error {{expected 'language', 'defined_in', or 'generated_declaration'}}

void f17()
__attribute__((external_source_symbol(language="Swift", "generated_declaration"))); // expected-error {{expected 'language', 'defined_in', or 'generated_declaration'}}

void f18()
__attribute__((external_source_symbol(language= =))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}

void f19()
__attribute__((external_source_symbol(defined_in="module" language="swift"))); // expected-error {{expected ')'}} expected-note {{to match this '('}}

void f20()
__attribute__((external_source_symbol(defined_in="module" language="swift" generated_declaration))); // expected-error {{expected ')'}} expected-note {{to match this '('}}

void f21()
__attribute__((external_source_symbol(defined_in= language="swift"))); // expected-error {{expected string literal for source container name in 'external_source_symbol' attribute}}

void f22()
__attribute__((external_source_symbol)); // expected-error {{'external_source_symbol' attribute takes at least 1 argument}}

void f23()
__attribute__((external_source_symbol(defined_in=, language="swift" generated_declaration))); // expected-error {{expected string literal for source container name in 'external_source_symbol' attribute}} expected-error{{expected ')'}} expected-note{{to match this '('}}

void f24()
__attribute__((external_source_symbol(language = generated_declaration))); // expected-error {{expected string literal for language name in 'external_source_symbol' attribute}}

void f25()
__attribute__((external_source_symbol(defined_in=123, defined_in="module"))); // expected-error {{expected string literal for source container name in 'external_source_symbol'}} expected-error {{duplicate 'defined_in' clause in an 'external_source_symbol' attribute}}

void f26()
__attribute__((external_source_symbol(language=Swift, language="Swift", error))); // expected-error {{expected string literal for language name in 'external_source_symbol'}} expected-error {{duplicate 'language' clause in an 'external_source_symbol' attribute}} expected-error {{expected 'language', 'defined_in', or 'generated_declaration'}}
