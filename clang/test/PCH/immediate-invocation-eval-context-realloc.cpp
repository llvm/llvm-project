// RUN: %clang_cc1 -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++20 -include-pch %t -verify %s
// RUN: %clang_cc1 -std=c++20 -include %s -verify %s

// expected-no-diagnostics

// GH232062: when an expression contains nested immediate invocations,
// HandleImmediateInvocations runs a TreeTransform (ComplexRemove) over it while
// holding references into Sema::ExprEvalContexts. Rebuilding the call to
// format() during that transform re-checks the constraint on AsStringView,
// which pushes enough nested evaluation contexts (about 3 per level of the
// C0..C9 chain) to reallocate ExprEvalContexts, leaving the references
// dangling.
//
// The PCH matters: in the main file, Sema starts with a fresh ExprEvalContexts
// (8 elements inline), and the rebuild repeats the full constraint check. The
// static_assert on P2 in the main file first moves the vector to a heap buffer
// of 17 elements, so that the reallocation during the rebuild (to more than 17
// elements) frees the memory the dangling references point into. Without that,
// the stale references would point into the never-freed inline storage, and
// the bug would go unnoticed even under ASan.
//
// The depths are chosen with margin: in the main file, the static_assert peaks
// at about 11 contexts (between 9 and 17, so the vector moves to a 17-element
// heap buffer), and the rebuild peaks at about 32 (more than 17).
//
// Without a fix, the freed buffer gets reused during the rebuild in practice,
// and the compile crashes. Strictly, that depends on the allocator; ASan
// detects the use-after-free directly.

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

template <class T> concept C0 = requires(T &t) { t = t; };
template <class T> concept C1 = requires { requires C0<T>; };
template <class T> concept C2 = requires { requires C1<T>; };
template <class T> concept C3 = requires { requires C2<T>; };
template <class T> concept C4 = requires { requires C3<T>; };
template <class T> concept C5 = requires { requires C4<T>; };
template <class T> concept C6 = requires { requires C5<T>; };
template <class T> concept C7 = requires { requires C6<T>; };
template <class T> concept C8 = requires { requires C7<T>; };
template <class T> concept C9 = requires { requires C8<T>; };

template <class> struct FormatString {
  consteval FormatString(int) {}
};
consteval int fmt(int) { return 0; }

template <C9> using AsStringView = int;
template <class S> void format(FormatString<AsStringView<S>>, S);

void emplace_test(auto) { format(fmt(0), 0); }

#else

template <class T> concept P0 = requires(T &t) { t = t; };
template <class T> concept P1 = requires { requires P0<T>; };
template <class T> concept P2 = requires { requires P1<T>; };
static_assert(P2<long>);

void f() { emplace_test(0); }

#endif
