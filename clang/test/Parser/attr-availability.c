// RUN: %clang_cc1 -std=c11 -fsyntax-only -verify %s

#if !__has_feature(attribute_availability)
#  error 'availability' attribute is not available
#endif

void f0(void) __attribute__((availability(macosx,introduced=10.2,deprecated=10.4,obsoleted=10.6)));

void f1(void) __attribute__((availability(macosx,deprecated=10.4,introduced=10.2,obsoleted=10.6)));

void f2(void) __attribute__((availability(ios,deprecated=10.4.7,introduced=10,obsoleted=10.6)));

void f3(void) __attribute__((availability(ios,deprecated=10.4.7,introduced=10,obsoleted=10.6,introduced=10.2))); // expected-error{{redundant 'introduced' availability change; only the last specified change will be used}}

void f4(void) __attribute__((availability(macosx,introduced=10.5), availability(ios,unavailable)));

void f5(void) __attribute__((availability(macosx,introduced=10.5), availability(ios,unavailable, unavailable))); // expected-error{{redundant 'unavailable' availability change; only the last specified change will be used}}

void f6(void) __attribute__((availability(macosx,unavailable,introduced=10.5))); // expected-warning{{'unavailable' availability overrides all other availability information}}

void f7(void) __attribute__((availability(macosx,message=L"wide"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}

void f8(void) __attribute__((availability(macosx,message="a" L"b"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}

void f9(void) __attribute__((availability(macosx,message=u8"b"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}

void f10(void) __attribute__((availability(macosx,message="a" u8"b"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}

void f11(void) __attribute__((availability(macosx,message=u"b"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}

void f12(void) __attribute__((availability(macosx,message="a" u"b"))); // expected-error {{expected string literal for optional message in 'availability' attribute}}

enum E{
    gorf __attribute__((availability(macosx,introduced=8.5, message = 10.0))), // expected-error {{expected string literal for optional message in 'availability' attribute}}
    garf __attribute__((availability(macosx,introduced=8.5, message))), // expected-error {{expected '=' after 'message'}}

    foo __attribute__((availability(macosx,introduced=8.5,deprecated=9.0, message="Use CTFontCopyPostScriptName()", deprecated=10.0))) // expected-error {{expected ')'}} \
                                                                                                                                       // expected-note {{to match this '('}}
};
