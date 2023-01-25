// RUN: %clang_cc1 -fsyntax-only -verify -ftemplate-backtrace-limit 2 %s
//
// FIXME: Disable this test when Clang was built with ASan, because ASan
// increases our per-frame stack usage enough that this test no longer fits
// within our normal stack space allocation.
// UNSUPPORTED: asan
//
// The default stack size on NetBSD is too small for this test.
// UNSUPPORTED: system-netbsd

template<int N, typename T> struct X : X<N+1, T*> {};
// expected-error-re@11 {{recursive template instantiation exceeded maximum depth of 1024{{$}}}}
// expected-note@11 {{instantiation of template class}}
// expected-note@11 {{skipping 1023 contexts in backtrace}}
// expected-note@11 {{use -ftemplate-depth=N to increase recursive template instantiation depth}}

X<0, int> x; // expected-note {{in instantiation of}}

// FIXME: It crashes. Investigating.
// UNSUPPORTED: target={{.*-windows-gnu}}
