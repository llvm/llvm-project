// RUN: %clang_cc1 -ast-print -std=c++14 %s -o %t.1.cpp
// RUN: %clang_cc1 -ast-print -std=c++14 %t.1.cpp -o %t.2.cpp
// RUN: diff %t.1.cpp %t.2.cpp

template<typename T> void func_01();
template<typename T> void func_01() {}
template<> void func_01<int>() {}
template<> void func_01<long>() {}
template<typename T> void func_01();

void main_01() {
  func_01<int*>();
  func_01<char>();
}

template<typename T> void func_02();
template<typename T> void func_02();
template<> void func_02<int>();
template<> void func_02<long>();
template<typename T> void func_02();

void main_02() {
  func_02<int*>();
  func_02<char>();
}
