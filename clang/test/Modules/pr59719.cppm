// https://github.com/llvm/llvm-project/issues/59780
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/data.cppm -emit-module-interface -o %t/data.pcm
// RUN: %clang_cc1 -std=c++20 %t/main.cpp -fprebuilt-module-path=%t -fsyntax-only -verify

//--- foo.h
namespace std {

template <class _Tp>
class expected {
public:
  expected(_Tp&& __u)
    {}

   constexpr ~expected()
    requires(__is_trivially_destructible(_Tp))
  = default;

   constexpr ~expected()
    requires(!__is_trivially_destructible(_Tp))
  {
  }
};

template <class _Tp>
class unique_ptr {
public:
   unique_ptr(void* __p) {}
   ~unique_ptr() {}
};

}

//--- data.cppm
module;
#include "foo.h"
export module data;
export namespace std {
    using std::unique_ptr;
    using std::expected;                    
}

export std::expected<std::unique_ptr<int>> parse() {
  return std::unique_ptr<int>(nullptr);                                             
}

//--- main.cpp
// expected-no-diagnostics
import data;
                                                                                
int main(int argc, const char *argv[]) {                                        
  std::expected<std::unique_ptr<int>> result = parse();                    
}
