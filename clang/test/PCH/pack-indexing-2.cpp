// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++2c -x c++-header %t/GH172464.h -emit-pch -o %t/GH172464.pch
// RUN: %clang_cc1 -std=c++2c -x c++ %t/GH172464.cpp -include-pch %t/GH172464.pch

//--- GH172464.h
template <class... Ts> struct _TypeInfo {
	template <int id> using type = Ts...[id];
};
using TypeInfo = _TypeInfo<int>;

TypeInfo::type<0> a;

//--- GH172464.cpp
int main() {
  TypeInfo::type<0> a;
}
