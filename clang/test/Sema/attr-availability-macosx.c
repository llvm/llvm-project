// RUN: %clang_cc1 "-triple" "x86_64-apple-darwin9.0.0" -fsyntax-only -verify %s

#if !__has_feature(attribute_availability_with_strict)
#error "Missing __has_feature"
#endif

void f0(int) __attribute__((availability(macosx,introduced=10.4,deprecated=10.6)));
void f1(int) __attribute__((availability(macosx,introduced=10.5)));
void f2(int) __attribute__((availability(macosx,introduced=10.4,deprecated=10.5))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(macosx,introduced=10.6)));
void f4(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(ios,introduced=2.0,deprecated=3.0))); // expected-note{{explicitly marked unavailable}}
void f5(int) __attribute__((availability(ios,introduced=3.2), availability(macosx,unavailable))); // expected-note{{'f5' has been explicitly marked unavailable here}}
void f6(int) __attribute__((availability(macOS,strict,introduced=10.6))); //expected-note{{'f6' has been explicitly marked unavailable here}}

void test(void) {
  f0(0);
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in macOS 10.5}}
  f3(0);
  f4(0); // expected-error{{f4' is unavailable: obsoleted in macOS 10.5}}
  f5(0); // expected-error{{'f5' is unavailable: not available on macOS}}
  f6(0); // expected-error{{'f6' is unavailable: introduced in macOS 10.6}}
}

struct __attribute__((availability(macosx,strict,introduced=10.6)))
  not_yet_introduced_struct; // \
    expected-note{{'not_yet_introduced_struct' has been explicitly marked unavailable here}}

void uses_not_introduced_struct(struct not_yet_introduced_struct *); // \
    expected-error{{'not_yet_introduced_struct' is unavailable: introduced in macOS 10.6}}

__attribute__((availability(macosx,strict,introduced=10.6)))
void uses_not_introduced_struct_same_availability(struct not_yet_introduced_struct *);

// rdar://10535640

enum {
    foo __attribute__((availability(macosx,introduced=8.0,deprecated=9.0)))
};

enum {
    bar __attribute__((availability(macosx,introduced=8.0,deprecated=9.0))) = foo
};

enum __attribute__((availability(macosx,introduced=8.0,deprecated=9.0))) {
    bar1 = foo
};

// Make sure the note is on the declaration with the actual availability attributes.
struct __attribute__((availability(macOS,strict,introduced=10.9))) type_info // \
    expected-note{{'type_info' has been explicitly marked unavailable here}}
{
};
struct type_info;
int test2(void) {
  struct type_info *t; // expected-error{{'type_info' is unavailable: introduced in macOS 10.9}}
  return 0;
}
