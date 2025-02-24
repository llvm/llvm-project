// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 \
// RUN:   -emit-pch -o %t/ctudir/ctu-test-import-failure-import.cpp.ast %S/Inputs/ctu-test-import-failure-import.cpp
// RUN: cp %S/Inputs/ctu-test-import-failure-import.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -analyze \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -verify %s

// Check that importing this code does not cause crash.
// Import intentionally fails because mismatch of '__get_first_arg'.

namespace std {
inline namespace __cxx11 {}
template <typename _CharT, typename> class basic_istream;
struct __get_first_arg;
inline namespace __cxx11 {
template <typename, typename, typename> class basic_string;
}
template <typename _CharT, typename _Traits, typename _Alloc>
basic_istream<_CharT, _Traits> &getline(basic_istream<_CharT, _Traits> &,
                                        basic_string<_CharT, _Traits, _Alloc> &,
                                        _CharT) {}
} // namespace std
namespace CommandLine {
extern const int RootExamples[];
}

// expected-warning@Inputs/ctu-test-import-failure-import.cpp:14{{incompatible definitions}}
// expected-warning@Inputs/ctu-test-import-failure-import.cpp:14{{incompatible definitions}}
// expected-note@Inputs/ctu-test-import-failure-import.cpp:14{{no corresponding field here}}
// expected-note@Inputs/ctu-test-import-failure-import.cpp:14{{no corresponding field here}}
