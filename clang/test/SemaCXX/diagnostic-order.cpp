// RUN: not %clang_cc1 %s -fsyntax-only 2>&1 | FileCheck %s

// Ensure that the diagnostics we produce for this situation appear in a
// deterministic order. This requires ADL to provide lookup results in a
// deterministic order.
template<typename T> struct Error { typedef typename T::error error; };
struct X { template<typename T> friend typename Error<T>::error f(X, T); };
struct Y { template<typename T> friend typename Error<T>::error f(T, Y); };

void g() {
  f(X(), Y());
}

// We don't really care which order these two diagnostics appear (although the
// order below is source order, which seems best). The crucial fact is that
// there is one single order that is stable across multiple runs of clang.
//
// CHECK: no type named 'error' in 'Y'
// CHECK: no type named 'error' in 'X'
// CHECK: no matching function for call to 'f'
