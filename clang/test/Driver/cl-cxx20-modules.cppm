// REQUIRES: system-windows

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cl /std:c++20 --precompile "%/t/Hello.cppm" "/Fo%/t/Hello.pcm"
// RUN: %clang_cl /std:c++20 %/t/use.cpp -fmodule-file=Hello=%/t/Hello.pcm %/t/Hello.pcm /Fo%/t/SimpleHelloWorld.exe 2>&1

// RUN: %/t/SimpleHelloWorld.exe
// CHECK: Simple Hello World!

//--- Hello.cppm
module;
#include <iostream>
export module Hello;
export void hello() {
  std::cout << "Simple Hello World!\n";
}

//--- use.cpp
import Hello;
int main() {
  hello();
  return 0;
}

//--- M.cppm
export module M;
export import :interface_part;
import :impl_part;
export void Hello();

//--- interface_part.cpp
export module M:interface_part;
export void World();

//--- impl_part.cppm
module;
#include <iostream>
#include <string>
module M:impl_part;
import :interface_part;

std::string W = "World!";
void World() {
  std::cout << W << std::endl;
}

//--- Impl.cpp
module;
#include <iostream>
module M;
void Hello() {
  std::cout << "Complex Hello ";
}

//--- User.cpp
import M;
int main() {
  Hello();
  World();
  return 0;
}

