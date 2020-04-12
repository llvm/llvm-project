// RUN: %clang_cc1 -fsyntax-only -verify -fms-extensions %s -Wno-deprecated-declarations
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify -fms-extensions %s -Wno-deprecated-declarations

typedef struct _GUID {
  __UINT32_TYPE__ Data1;
  __UINT16_TYPE__ Data2;
  __UINT16_TYPE__ Data3;
  __UINT8_TYPE__ Data4[8];
} GUID;

namespace {
// cl.exe's behavior with merging uuid attributes is a bit erratic:
// * In []-style attributes, a single [] list must not list a duplicate uuid
//   (even if it's the same uuid), and only a single declaration of a class
//   must have a uuid else the compiler errors out (even if two declarations of
//   a class have the same uuid).
// * For __declspec(uuid(...)), it's ok if several declarations of a class have
//   an uuid, as long as it's the same uuid each time.  If uuids on declarations
//   don't match, the compiler errors out.
// * If there are several __declspec(uuid(...))s on one declaration, the
//   compiler only warns about this and uses the last uuid.  It even warns if
//   the uuids are the same.

// clang-cl implements the following simpler (but largely compatible) behavior
// instead:
// * [] and __declspec uuids have the same behavior.
// * If there are several uuids on a class (no matter if on the same decl or
//   on several decls), it is an error if they don't match.
// * Having several uuids that match is ok.

// Both cl and clang-cl accept this:
class __declspec(uuid("000000A0-0000-0000-C000-000000000049")) C1;
class __declspec(uuid("000000a0-0000-0000-c000-000000000049")) C1;
class __declspec(uuid("{000000a0-0000-0000-C000-000000000049}")) C1;
class __declspec(uuid("000000A0-0000-0000-C000-000000000049")) C1 {};

// Both cl and clang-cl error out on this:
// expected-note@+1 2{{previous uuid specified here}}
class __declspec(uuid("000000A0-0000-0000-C000-000000000049")) C2;
// expected-error@+1 {{uuid does not match previous declaration}}
class __declspec(uuid("110000A0-0000-0000-C000-000000000049")) C2;
// expected-error@+1 {{uuid does not match previous declaration}}
class __declspec(uuid("220000A0-0000-0000-C000-000000000049")) C2 {};

// expected-note@+1 {{previous uuid specified here}}
class __declspec(uuid("000000A0-0000-0000-C000-000000000049")) C2_2;
class C2_2;
// expected-error@+1 {{uuid does not match previous declaration}}
class __declspec(uuid("110000A0-0000-0000-C000-000000000049")) C2_2;

// clang-cl accepts this, but cl errors out:
[uuid("000000A0-0000-0000-C000-000000000049")] class C3;
[uuid("000000A0-0000-0000-C000-000000000049")] class C3;
[uuid("000000A0-0000-0000-C000-000000000049")] class C3 {};

// Both cl and clang-cl error out on this (but for different reasons):
// expected-note@+1 2{{previous uuid specified here}}
[uuid("000000A0-0000-0000-C000-000000000049")] class C4;
// expected-error@+1 {{uuid does not match previous declaration}}
[uuid("110000A0-0000-0000-C000-000000000049")] class C4;
// expected-error@+1 {{uuid does not match previous declaration}}
[uuid("220000A0-0000-0000-C000-000000000049")] class C4 {};

// Both cl and clang-cl error out on this:
// expected-error@+1 {{uuid does not match previous declaration}}
class __declspec(uuid("000000A0-0000-0000-C000-000000000049"))
// expected-note@+1 {{previous uuid specified here}}
      __declspec(uuid("110000A0-0000-0000-C000-000000000049")) C5;

// expected-error@+1 {{uuid does not match previous declaration}}
[uuid("000000A0-0000-0000-C000-000000000049"),
// expected-note@+1 {{previous uuid specified here}}
 uuid("110000A0-0000-0000-C000-000000000049")] class C6;

// cl doesn't diagnose having one uuid each as []-style attributes and as
// __declspec, even if the uuids differ.  clang-cl errors if they differ.
[uuid("000000A0-0000-0000-C000-000000000049")]
class __declspec(uuid("000000A0-0000-0000-C000-000000000049")) C7;

// expected-note@+1 {{previous uuid specified here}}
[uuid("000000A0-0000-0000-C000-000000000049")]
// expected-error@+1 {{uuid does not match previous declaration}}
class __declspec(uuid("110000A0-0000-0000-C000-000000000049")) C8;


// cl warns on this, but clang-cl is fine with it (which is consistent with
// e.g. specifying __multiple_inheritance several times, which cl accepts
// without warning too).
class __declspec(uuid("000000A0-0000-0000-C000-000000000049"))
      __declspec(uuid("000000A0-0000-0000-C000-000000000049")) C9;

// cl errors out on this, but clang-cl is fine with it (to be consistent with
// the previous case).
[uuid("000000A0-0000-0000-C000-000000000049"),
 uuid("000000A0-0000-0000-C000-000000000049")] class C10;

template <const GUID* p>
void F1() {
  // Regression test for PR24986. The given GUID should just work as a pointer.
  const GUID* q = p;
}

void F2() {
  // The UUID should work for a non-type template parameter.
  F1<&__uuidof(C1)>();
}

}

// Test class/struct redeclaration where the subsequent
// declaration has a uuid attribute
struct X{};

struct __declspec(uuid("00000000-0000-0000-0000-000000000000")) X;

namespace ConstantEvaluation {
  class __declspec(uuid("1babb1ed-feed-c01d-1ced-decafc0ffee5")) Request;
  constexpr GUID a = __uuidof(Request);
  static_assert(a.Data1 == 0x1babb1ed, "");
  static_assert(__uuidof(Request).Data1 == 0x1babb1ed, "");
  static_assert(a.Data2 == 0xfeed, "");
  static_assert(__uuidof(Request).Data2 == 0xfeed, "");
  static_assert(a.Data3 == 0xc01d, "");
  static_assert(__uuidof(Request).Data3 == 0xc01d, "");
  static_assert(a.Data4[0] == 0x1c, "");
  static_assert(__uuidof(Request).Data4[0] == 0x1c, "");
  static_assert(a.Data4[1] == 0xed, "");
  static_assert(__uuidof(Request).Data4[1] == 0xed, "");
  static_assert(a.Data4[2] == 0xde, "");
  static_assert(__uuidof(Request).Data4[2] == 0xde, "");
  static_assert(a.Data4[7] == 0xe5, "");
  static_assert(__uuidof(Request).Data4[7] == 0xe5, "");
  constexpr int k = __uuidof(Request).Data4[8]; // expected-error {{constant expression}} expected-note {{past-the-end}}
}
