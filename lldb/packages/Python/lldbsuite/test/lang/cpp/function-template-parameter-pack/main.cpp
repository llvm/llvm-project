//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LIDENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

template <class T> int staticSizeof() {
  return sizeof(T);
}

template <class T1, class T2, class... Ts> int staticSizeof() {
  return staticSizeof<T2, Ts...>() + sizeof(T1);
}

int main (int argc, char const *argv[])
{
  int sz = staticSizeof<long, int, char>();
  return staticSizeof<long, int, char>() != sz; //% self.expect("expression -- sz == staticSizeof<long, int, char>()", "staticSizeof<long, int, char> worked", substrs = ["true"])
                                  //% self.expect("expression -- sz == staticSizeof<long, int>() + sizeof(char)", "staticSizeof<long, int> worked", substrs = ["true"])
                                  //% self.expect("expression -- sz == staticSizeof<long>() + sizeof(int) + sizeof(char)", "staticSizeof<long> worked", substrs = ["true"])
}
