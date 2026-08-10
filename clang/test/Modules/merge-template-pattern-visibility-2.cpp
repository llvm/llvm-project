// RUN: %clang_cc1 -fmodules -fmodules-local-submodule-visibility %s -verify -Werror=undefined-inline

#pragma clang module build A1
module A1 { export * }
#pragma clang module contents
#pragma clang module begin A1
template<typename T> class A {};
template<typename T> inline bool f(const A<T>&) { return T::error; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build A2
module A2 { export * }
#pragma clang module contents
#pragma clang module begin A2
#pragma clang module load A1
template<typename T> class A {};
template<typename T> inline bool f(const A<T>&) { return T::error; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module build A3
module A3 { export * }
#pragma clang module contents
#pragma clang module begin A3
template<typename T> class A {};
template<typename T> inline bool f(const A<T>&) { return T::error; }
#pragma clang module end
#pragma clang module endbuild

#pragma clang module load A3
#pragma clang module import A2
// expected-error@* {{cannot be used prior to}}
bool y(A<int> o) { return f(o); } // expected-note {{instantiation of}}
