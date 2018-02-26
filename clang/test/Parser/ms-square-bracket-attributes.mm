// RUN: %clang_cc1 -fsyntax-only -std=c++14 -verify -fms-extensions %s -Wno-deprecated-declarations

typedef struct _GUID {
  unsigned long Data1;
  unsigned short Data2;
  unsigned short Data3;
  unsigned char Data4[8];
} GUID;

namespace {
// cl.exe supports [] attributes on decls like so:
[uuid( "000000A0-0000-0000-C000-000000000049" )] struct struct_with_uuid;

// Optionally, the uuid can be surrounded by one set of braces.
[uuid(
  "{000000A0-0000-0000-C000-000000000049}"
)] struct struct_with_uuid_brace;

// uuids must be ascii string literals.
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid(u8"000000A0-0000-0000-C000-000000000049")] struct struct_with_uuid_u8;
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid(L"000000A0-0000-0000-C000-000000000049")] struct struct_with_uuid_L;

// cl.exe doesn't allow raw string literals in []-style attributes, but does
// allow it for __declspec(uuid()) (u8 literals etc are not allowed there
// either).  Since raw string literals not being allowed seems like an
// implementation artifact in cl and not allowing them makes the parse code
// a bit unnatural, do allow this.
[uuid(R"(000000A0-0000-0000-C000-000000000049)")] struct struct_with_uuid_raw;

// Likewise, cl supports UCNs in declspec uuid, but not in []-style uuid.
// clang-cl allows them in both.
[uuid("000000A0-0000\u002D0000-C000-000000000049")] struct struct_with_uuid_ucn;

// cl doesn't allow string concatenation in []-style attributes, for no good
// reason.  clang-cl allows them.
[uuid("000000A0-00" "00-0000-C000-000000000049")] struct struct_with_uuid_split;

// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
[uuid("{000000A0-0000-0000-C000-000000000049}", "1")] struct S {};
// expected-error@+1 {{expected '('}}
[uuid{"000000A0-0000-0000-C000-000000000049"}] struct T {};
// expected-error@+1 {{expected ')'}} expected-note@+1 {{to match this '('}}
[uuid("000000A0-0000-0000-C000-000000000049"}] struct U {};


// In addition to uuids in string literals, cl also allows uuids that are not
// in a string literal, only delimited by ().  The contents of () are almost
// treated like a literal (spaces there aren't ignored), but macro substitution,
// \ newline escapes, and so on are performed.

[ uuid (000000A0-0000-0000-C000-000000000049) ] struct struct_with_uuid2;
[uuid({000000A0-0000-0000-C000-000000000049})] struct struct_with_uuid2_brace;

// The non-quoted form doesn't allow any whitespace inside the parens:
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid( 000000A0-0000-0000-C000-000000000049)] struct struct_with_uuid2;
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid(000000A0-0000 -0000-C000-000000000049)] struct struct_with_uuid2;
// expected-error@+2 {{uuid attribute contains a malformed GUID}}
[uuid(000000A0-0000
-0000-C000-000000000049)] struct struct_with_uuid2;
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid(000000A0-0000/**/-0000-C000-000000000049)] struct struct_with_uuid2;
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid(000000A0-0000-0000-C000-000000000049 )] struct struct_with_uuid2;
// expected-error@+2 {{uuid attribute contains a malformed GUID}}
[uuid(000000A0-0000-0000-C000-000000000049
)
] struct struct_with_uuid2;
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid({000000A0-0000-""0000-C000-000000000049})] struct struct_with_uuid2;

// Line continuations and macro substitution are fine though:
[uuid(000000A0-0000-0000-\
C000-000000000049)] struct struct_with_uuid2_cont;
#define UUID 000000A0-0000-0000-C000-000000000049
#define UUID_PART 000000A0-0000
[uuid(UUID)] struct struct_with_uuid2_macro;
[uuid(UUID_PART-0000-C000-000000000049)] struct struct_with_uuid2_macro_part;

// Both cl and clang-cl accept trigraphs here (with /Zc:trigraphs, off by
// default)
// expected-warning@+1 2{{trigraph converted}}
[uuid(??<000000A0-0000-0000-C000-000000000049??>)]
struct struct_with_uuid2_trigraph;

// UCNs cannot be used in this form because they're prohibited by C99.
// expected-error@+1 {{character '-' cannot be specified by a universal character name}}
[uuid(000000A0-0000\u002D0000-C000-000000000049)] struct struct_with_uuid2_ucn;

// Invalid digits.
// expected-error@+1 {{uuid attribute contains a malformed GUID}}
[uuid(0Z0000A0-0000-0000-C000-000000000049)] struct struct_with_uuid2;

struct OuterClass {
  // [] uuids and inner classes are weird in cl.exe: It warns that uuid on
  // nested types has undefined behavior, and errors out __uuidof() claiming
  // that the inner type has no assigned uuid.  Things work fine if __declspec()
  // is used instead.  clang-cl handles this fine.
  [uuid(10000000-0000-0000-0000-000000000000)] class InnerClass1;
  [uuid(10000000-0000-0000-0000-000000000000)] class InnerClass2 {} ic;
  [uuid(10000000-0000-0000-0000-000000000000)] static class InnerClass3 {} sic;
  // Putting `static` in front of [...] causes parse errors in both cl and clang

  // This is the only syntax to declare an inner class with []-style attributes
  // that works in cl: Declare the inner class without an attribute, and then
  // have the []-style attribute on the definition.
  class InnerClass;
};
[uuid(10000000-0000-0000-0000-000000000000)] class OuterClass::InnerClass {};

void use_it() {
  (void)__uuidof(struct_with_uuid);
  (void)__uuidof(struct_with_uuid_brace);
  (void)__uuidof(struct_with_uuid_raw);
  (void)__uuidof(struct_with_uuid_ucn);
  (void)__uuidof(struct_with_uuid_split);

  (void)__uuidof(struct_with_uuid2);
  (void)__uuidof(struct_with_uuid2_brace);
  (void)__uuidof(struct_with_uuid2_cont);
  (void)__uuidof(struct_with_uuid2_macro);
  (void)__uuidof(struct_with_uuid2_macro_part);
  (void)__uuidof(struct_with_uuid2_trigraph);

  (void)__uuidof(OuterClass::InnerClass);
  (void)__uuidof(OuterClass::InnerClass1);
  (void)__uuidof(OuterClass::InnerClass2);
  (void)__uuidof(OuterClass::InnerClass3);
  (void)__uuidof(OuterClass().ic);
  (void)__uuidof(OuterClass::sic);
}

// expected-warning@+1 {{'uuid' attribute only applies to structs, unions, classes, and enums}}
[uuid("000000A0-0000-0000-C000-000000000049")] void f();
}

// clang supports these on toplevel decls, but not on local decls since this
// syntax is ambiguous with lambdas and Objective-C message send expressions.
// This file documents clang's shortcomings and lists a few constructs that
// one has to keep in mind when trying to fix this.  System headers only seem
// to use these attributes on toplevel decls, so supporting this is not very
// important.

void local_class() {
  // FIXME: MSVC accepts, but we reject due to ambiguity.
  // expected-error@+1 {{expected body of lambda expression}}
  [uuid("a5a7bd07-3b14-49bc-9399-de066d4d72cd")] struct Local {
    int x;
  };
}

void useit(int);
int lambda() {
  int uuid = 42;
  [uuid]() { useit(uuid); }();

  // C++14 lambda init captures:
  [uuid(00000000-0000-0000-0000-000000000000)] { return uuid; }();
  [uuid("00000000-0000-0000-0000-000000000000")](int n) { return uuid[n]; }(3);
}

@interface NSObject
- (void)retain;
@end
int message_send(id uuid) {
  [uuid retain]; 
}
NSObject* uuid(const char*);
int message_send2() {
  [uuid("a5a7bd07-3b14-49bc-9399-de066d4d72cd") retain]; 
}
