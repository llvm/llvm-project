// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1                                         -verify=both,ref      %s

// ref-no-diagnostics

typedef __SIZE_TYPE__ size_t;

int a;
static_assert(__builtin_object_size(&a, 0) == sizeof(int), "");
float f;
static_assert(__builtin_object_size(&f, 0) == sizeof(float), "");
int arr[2];
static_assert(__builtin_object_size(&arr, 0) == (sizeof(int)*2), "");

float arrf[2];
static_assert(__builtin_object_size(&arrf, 0) == (sizeof(float)*2), "");
static_assert(__builtin_object_size(&arrf[1], 0) == sizeof(float), "");
static_assert(__builtin_object_size(&arrf[2], 0) == 0, "");

constexpr struct { int a; int b; } F{};
static_assert(__builtin_object_size(&F.a, 3) == sizeof(int));

struct S {
  int a;
  char c;
};

S s;
static_assert(__builtin_object_size(&s, 0) == sizeof(s), "");

S ss[2];
static_assert(__builtin_object_size(&ss[1], 0) == sizeof(s), "");
static_assert(__builtin_object_size(&ss[1].c, 0) == sizeof(int), "");

struct A { char buf[16]; };
struct B : A {};
struct C { int i; B bs[1]; } c;
static_assert(__builtin_object_size(&c.bs[0], 3) == 16);
static_assert(__builtin_object_size(&c.bs[1], 3) == 0);

/// These are from test/SemaCXX/builtin-object-size-cxx14.
/// They all don't work since they are rejected when evaluating the first
/// parameter of the __builtin_object_size call.
///
/// GCC rejects them as well.
namespace InvalidBase {
  // Ensure this doesn't crash.
  struct S { const char *name; };
  S invalid_base(); // expected-note {{declared here}}
  constexpr size_t bos_name = __builtin_object_size(invalid_base().name, 1); // expected-error {{must be initialized by a constant expression}} \
                                                                             // expected-note {{non-constexpr function 'invalid_base'}}

  struct T { ~T(); };
  T invalid_base_2();
  constexpr size_t bos_dtor = __builtin_object_size(&(T&)(T&&)invalid_base_2(), 0); // expected-error {{must be initialized by a constant expression}} \
                                                                                    // expected-note {{non-literal type 'T'}}
}
