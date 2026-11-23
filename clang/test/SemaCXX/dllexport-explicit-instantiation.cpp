// RUN: %clang_cc1 -triple x86_64-win32   -fsyntax-only -fms-extensions -verify=win32,win32-pedantic %s -Wdllexport-explicit-instantiation
// RUN: %clang_cc1 -triple x86_64-win32   -fsyntax-only -fms-extensions -verify=win32                %s -Wno-dllexport-explicit-instantiation
// RUN: %clang_cc1 -triple x86_64-mingw32 -fsyntax-only -fms-extensions -verify=mingw,mingw-pedantic %s -Wdllexport-explicit-instantiation
// RUN: %clang_cc1 -triple x86_64-mingw32 -fsyntax-only -fms-extensions -verify=mingw                %s -Wno-dllexport-explicit-instantiation

template <class>
class S {};

extern template class __declspec(dllexport) S<short>; // win32-pedantic-warning {{explicit instantiation declaration should not be 'dllexport'}} \
                                                         win32-pedantic-note {{attribute is here}}
template class __declspec(dllexport) S<short>;        // mingw-pedantic-warning {{'dllexport' attribute ignored on explicit instantiation definition}}

extern template class __declspec(dllexport) S<int>;  // win32-pedantic-warning {{explicit instantiation declaration should not be 'dllexport'}} \
                                                        win32-pedantic-note {{attribute is here}} \
                                                        win32-note {{'dllexport' attribute on the declaration is ignored}}
template class S<int>;                               // win32-warning {{explicit instantiation definition is not exported without 'dllexport'}}

extern template class S<long>;                       // mingw-note {{'dllexport' attribute is missing on previous declaration}}
template class __declspec(dllexport) S<long>;        // mingw-warning {{'dllexport' attribute ignored on explicit instantiation definition}}
