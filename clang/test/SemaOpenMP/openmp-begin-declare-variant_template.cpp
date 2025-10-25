// RUN: %clang_cc1 -triple x86_64 -fopenmp -verify %s

// FIXME: Is this supposed to work?

#pragma omp begin declare variant match(implementation={extension(allow_templates)})
template <class T> void f(T) {}
// expected-note@-1 {{explicit instantiation refers here}}
#pragma end
template <int> struct A {};
template <bool B> A<B> f() = delete;
template void f<float>(float);
// expected-error@-1 {{explicit instantiation of undefined function template 'f'}}
