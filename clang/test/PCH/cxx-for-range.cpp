// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++11 -include %S/cxx-for-range.h -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++11 -emit-pch -o %t %S/cxx-for-range.h
// RUN: %clang_cc1 -x c++ -std=c++11 -include-pch %t -emit-llvm -o - %s 

void h() {
  f();

  g<int>();

  char a[3] = { 0, 1, 2 };
  for (auto w : a)
    for (auto x : S())
      for (auto y : T())
        for (auto z : U())
          ;
}
