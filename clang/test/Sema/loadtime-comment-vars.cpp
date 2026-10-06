// Verify that -mloadtime-comment-vars= diagnoses C++ variables it cannot
// preserve.
//
// Three scenarios are covered, each with its own set of named variables and
// its own -verify prefix:
//
//   storage — storage duration: thread_local variables, including a
//             thread_local static data member, are diagnosed; a
//             namespace-scope variable is accepted without diagnostic.
//   init    — initializer form: dynamically initialized pointers and
//             arrays, and constant initializers that are not a direct string literal
//             (pointer to another global, consteval call, user-defined
//             literal), are diagnosed. (-std=c++20 for consteval.)
//   kinds   — unsupported declaration kinds and types: a name-matched
//             function-local static, static data members (out-of-line,
//             in-class inline, instantiated from a class template), variable
//             template specializations (explicit and implicit),
//             namespace-scope inline variables, and a variable of an
//             unsupported (wide character) type are diagnosed.

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=keep,_ZN1N2tlE,_ZL3stl,_ZN1A2tmE \
// RUN:   -fsyntax-only -verify=storage %s

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=p_ok,arr_ok,p_dyn,arr_dyn,p_ind,p_ce,p_udl \
// RUN:   -fsyntax-only -verify=init %s

// RUN: %clang_cc1 -std=c++20 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=_ZZ1fvE2fn,_ZN2B23sidE,_ZN2B34isidE,_Z2vtIcE,_Z2vtIiE,_ZN2SCIiE1mE,_ZN1M2ivE,_ZL4wstr \
// RUN:   -fsyntax-only -verify=kinds %s

// ---- storage: storage-duration cases ----------------------------------------

// A namespace-scope variable with static storage duration is supported; no
// diagnostic is expected.
const char *keep = "@(#) keep";

namespace N {
// A thread_local variable has thread storage duration rather than static,
// so it is diagnosed.
thread_local const char *tl = "@(#) tl"; // storage-warning {{'tl' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}
} // namespace N

// 'static' here changes linkage only; storage duration is still thread.
static thread_local const char *stl = "@(#) stl"; // storage-warning {{'stl' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}

struct A {
  static thread_local const char *tm;
};
// The storage-duration reason outranks the static-data-member one.
thread_local const char *A::tm = "@(#) tm"; // storage-warning {{'tm' named in '-mloadtime-comment-vars=' does not have static storage duration and will not be preserved}}

// ---- init: initializer-form cases --------------------------------------------

const char *make();
char make_char();

// A pointer bound directly to a string literal is supported; no diagnostic
// is expected.
const char *p_ok = "@(#) p_ok";

// An array initialized from a string literal is supported; no diagnostic is
// expected.
char arr_ok[] = "@(#) arr_ok";

// A constant character array referenced by the pointer below.
const char src[] = "@(#) src";

// Dynamic initialization: value set by a start-up constructor; the string
// would not be present in the object file at load time.
const char *p_dyn = make(); // init-warning {{'p_dyn' named in '-mloadtime-comment-vars=' is not constant-initialized and will not be preserved}}

// An array whose elements come from a non-constant call is dynamically
// initialized as well, and is diagnosed for the same reason.
char arr_dyn[] = {make_char(), '\0'}; // init-warning {{'arr_dyn' named in '-mloadtime-comment-vars=' is not constant-initialized and will not be preserved}}

// This pointer is constant-initialized but bound to another global rather
// than a string literal, so it is diagnosed.
const char *p_ind = src; // init-warning {{'p_ind' named in '-mloadtime-comment-vars=' is not initialized with a string literal and will not be preserved}}

// Constant-initialized via an immediate (consteval) call. The initializer is
// a call, not a direct string literal — the pointed-to string is not
// guaranteed to be in this object file — so it is diagnosed, not preserved.
consteval const char *make_ce() { return "@(#) ce"; }
const char *p_ce = make_ce(); // init-warning {{'p_ce' named in '-mloadtime-comment-vars=' is not initialized with a string literal and will not be preserved}}

// A user-defined literal is likewise a call underneath: constant-initialized
// but not a direct string literal, so it is diagnosed.
typedef __SIZE_TYPE__ size_t;
constexpr const char *operator""_id(const char *str, size_t) { return str; }
const char *p_udl = "@(#) udl"_id; // init-warning {{'p_udl' named in '-mloadtime-comment-vars=' is not initialized with a string literal and will not be preserved}}

// ---- kinds: unsupported declaration kinds and types --------------------------

// Function-local static: a name match demonstrates intent, so it is
// diagnosed rather than silently ignored.
void f() { static const char *fn = "@(#) fn"; (void)fn; } // kinds-warning {{'fn' named in '-mloadtime-comment-vars=' is a function-local variable and will not be preserved}}

// A name match on a variable of an unsupported type is diagnosed as well.
static wchar_t wstr[] = L"@(#) w"; // kinds-warning {{'wstr' named in '-mloadtime-comment-vars=' does not have a plain char pointer or array type and will not be preserved}}

// An out-of-line static data member definition is diagnosed.
struct B2 {
  static const char *sid;
};
const char *B2::sid = "@(#) b2"; // kinds-warning {{'sid' named in '-mloadtime-comment-vars=' is a static data member and will not be preserved}}

// A namespace-scope inline variable is diagnosed.
namespace M {
inline const char *iv = "@(#) inline var"; // kinds-warning {{'iv' named in '-mloadtime-comment-vars=' is an inline variable and will not be preserved}}
} // namespace M

// An in-class inline static data member (C++17) is diagnosed as well.
struct B3 {
  static inline const char *isid = "@(#) b3"; // kinds-warning {{'isid' named in '-mloadtime-comment-vars=' is a static data member and will not be preserved}}
};

// Variable template: the explicit specialization is diagnosed at its own
// definition; the implicit specialization is diagnosed at the pattern, in
// the TU that instantiates it, with a note at the point of instantiation.
template <class T> const char *vt = "@(#) vt"; // kinds-warning {{'vt<int>' named in '-mloadtime-comment-vars=' is a variable template specialization and will not be preserved}}
template <> const char *vt<char> = "@(#) vtc"; // kinds-warning {{'vt<char>' named in '-mloadtime-comment-vars=' is a variable template specialization and will not be preserved}}
const char *use_vt = vt<int>; // kinds-note {{in instantiation of variable template specialization 'vt<int>' requested here}}

// Static data member instantiated from a class template — diagnosed at the
// member's pattern definition when the specialization is instantiated.
template <class T> struct SC {
  static const char *m;
};
template <class T> const char *SC<T>::m = "@(#) scm"; // kinds-warning {{'m' named in '-mloadtime-comment-vars=' is a static data member and will not be preserved}}
const char *use_scm = SC<int>::m; // kinds-note {{in instantiation of static data member 'SC<int>::m' requested here}}
