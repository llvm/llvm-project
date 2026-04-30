// RUN: %clang_cc1 -std=c++20 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify -verify-directives
// RUN: %clang_cc1 -std=c++23 -fexceptions -fcxx-exceptions -pedantic-errors %s -verify -verify-directives
// RUN: %clang_cc1 -std=c++2c -fexceptions -fcxx-exceptions -pedantic-errors %s -verify -verify-directives

export module foo;

namespace cwg2443 { // cwg2443: 23

export template <typename T> class s1 {};
export template <typename T> class s1<T *> {};
// expected-warning@-1 {{a specialization cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export template <> class s1<int> {};
// expected-warning@-1 {{a specialization cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export template class s1<char>;
// expected-warning@-1 {{an explicit instantiation cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export extern template class s1<void>;
// expected-warning@-1 {{an explicit instantiation cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}

export template <typename T> int v1 = 0;
export template <typename T> int v1<T *> = 0;
// expected-warning@-1 {{a specialization cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export template <> int v1<int> = 0;
// expected-warning@-1 {{a specialization cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export template int v1<char>;
// expected-warning@-1 {{an explicit instantiation cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export extern template int v1<void>;
// expected-warning@-1 {{an explicit instantiation cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}

export template <typename T> void f1() {}
export template <> void f1<int>() {}
// expected-warning@-1 {{a specialization cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export template void f1<char>();
// expected-warning@-1 {{an explicit instantiation cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}
export extern template void f1<void>();
// expected-warning@-1 {{an explicit instantiation cannot be marked 'export'}}
//   expected-note@-2 {{it is exported if the primary template is exported}}


export { template <typename T> class s2 {}; }
export { template <typename T> class s2<T *> {}; }
export { template <> class s2<int> {}; }
export { template class s2<char>; }
export { extern template class s2<void>; }

export { template <typename T> int v2 = 0; }
export { template <typename T> int v2<T *> = 0; }
export { template <> int v2<int> = 0; }
export { template int v2<char>; }
export { extern template int v2<void>; }

export { template <typename T> void f2() {} }
export { template <> void f2<int>() {} }
export { template void f2<char>(); }
export { extern template void f2<void>(); }

} // namespace cwg2443
