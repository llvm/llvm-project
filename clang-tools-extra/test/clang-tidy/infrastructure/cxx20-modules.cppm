// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
// RUN: mkdir %t/tmp
//
// RUN: %check_clang_tidy -std=c++20 -check-suffix=DEFAULT %t/a.cpp \
// RUN:   cppcoreguidelines-narrowing-conversions %t/a.cpp -- \
// RUN:   -config='{}'

// RUN: %clang -std=c++20 -x c++-module %t/a.cpp --precompile -o %t/a.pcm

// RUN: %check_clang_tidy -std=c++20 -check-suffix=DEFAULT %t/use.cpp \
// RUN:   cppcoreguidelines-narrowing-conversions %t/a.cpp -- \
// RUN:   -config='{}' -- -fmodule-file=a=%t/a.pcm 

//--- a.cpp
export module a;
export void most_narrowing_is_not_ok() {
  int i;
  long long ui;
  i = ui;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}

//--- use.cpp
import a;
void use() {
  most_narrowing_is_not_ok();
}
