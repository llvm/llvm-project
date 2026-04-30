// RUN: rm -rf %t
// RUN: split-file %s %t


// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only %t/A.cpp
// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only %t/B.cpp
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fdiagnostics-parseable-fixits %t/B.cpp 2>&1 | FileCheck %t/B.cpp
// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only %t/msvc-stl-exception-1.cpp
// RUN: %clang_cc1 -std=c++20 -verify -fsyntax-only %t/msvc-stl-exception-2.cpp

//--- A.cpp
// expected-no-diagnostics
export module A;
export namespace N {int x = 42;}
export using namespace N;

//--- B.cpp
export module B;

export template <typename T> class s1 {};
export template <typename T> class s1<T *> {}; // expected-warning {{a specialization cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export template <> class s1<int> {}; // expected-warning {{a specialization cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export template class s1<char>; // expected-warning {{an explicit instantiation cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export extern template class s1<void>; // expected-warning {{an explicit instantiation cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""

export template <typename T> int v1 = 0;
export template <typename T> int v1<T *> = 0; // expected-warning {{a specialization cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export template <> int v1<int> = 0; // expected-warning {{a specialization cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export template int v1<char>; // expected-warning {{an explicit instantiation cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export extern template int v1<void>; // expected-warning {{an explicit instantiation cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""

export template <typename T> void f1() {}
export template <> void f1<int>() {} // expected-warning {{a specialization cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export template void f1<char>(); // expected-warning {{an explicit instantiation cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""
export extern template void f1<void>(); // expected-warning {{an explicit instantiation cannot be marked 'export'}}
// expected-note@-1 {{it is exported if the primary template is exported}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:8}:""


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


extern "C++" template <typename T> class s3 {};
extern "C++" template <typename T> class s3<T *> {};
extern "C++" template <> class s3<int> {}; // expected-warning {{language linkage cannot be specified for a specialization}}
extern "C++" template class s3<char>; // expected-warning {{language linkage cannot be specified for an explicit instantiation}}
extern "C++" extern template class s3<void>; // expected-warning {{language linkage cannot be specified for an explicit instantiation}}

extern "C++" template <typename T> int v3 = 0;
extern "C++" template <typename T> int v3<T *> = 0;
extern "C++" template <> int v3<int> = 0; // expected-warning {{language linkage cannot be specified for a specialization}}
extern "C++" template int v3<char>; // expected-warning {{language linkage cannot be specified for an explicit instantiation}}
extern "C++" extern template int v3<void>; // expected-warning {{language linkage cannot be specified for an explicit instantiation}}

extern "C++" template <typename T> void f3() {}
extern "C++" template <> void f3<int>() {} // expected-warning {{language linkage cannot be specified for a specialization}}
extern "C++" template void f3<char>(); // expected-warning {{language linkage cannot be specified for an explicit instantiation}}
extern "C++" extern template void f3<void>(); // expected-warning {{language linkage cannot be specified for an explicit instantiation}}

extern "C++" export int i; // expected-warning {{language linkage cannot be specified for an export declaration}}
extern "C++" export {} // expected-warning {{language linkage cannot be specified for an export declaration}}


extern "C++" { template <typename T> class s4 {}; }
extern "C++" { template <typename T> class s4<T *> {}; }
extern "C++" { template <> class s4<int> {}; }
extern "C++" { template class s4<char>; }
extern "C++" { extern template class s4<void>; }

extern "C++" { template <typename T> int v4 = 0; }
extern "C++" { template <typename T> int v4<T *> = 0; }
extern "C++" { template <> int v4<int> = 0; }
extern "C++" { template int v4<char>; }
extern "C++" { extern template int v4<void>; }

extern "C++" { template <typename T> void f4() {} }
extern "C++" { template <> void f4<int>() {} }
extern "C++" { template void f4<char>(); }
extern "C++" { extern template void f4<void>(); }

//--- msvc-stl-exception-1.cpp
#define _MSVC_STL_UPDATE 202602L

namespace std {

extern "C++" template <typename T> struct s {};
extern "C++" template<> struct s<int> {};

} // namespace std

extern "C++" template <typename T> struct c {};
extern "C++" template<> struct c<int> {}; // expected-warning {{language linkage cannot be specified for a specialization}}

//--- msvc-stl-exception-2.cpp
#define _MSVC_STL_UPDATE 202603L

namespace std {

extern "C++" template <typename T> struct s {};
extern "C++" template<> struct s<int> {}; // expected-warning {{language linkage cannot be specified for a specialization}}

} // namespace std
