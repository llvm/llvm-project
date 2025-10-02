// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -verify -fsyntax-only

//--- bar.h
template <typename T>
struct Singleton {
    static T* instance_;
    static T* get() {
        static bool init = false;
        if (!init) {
            init = true;
            instance_ = ::new T();
        }
        return instance_;
    }
};

template <typename T>
T* Singleton<T>::instance_ = nullptr;

struct s{};
inline void* foo() {
    return Singleton<s>::get();
}

//--- foo.cppm
// expected-no-diagnostics
module;
#include "bar.h"
export module foo;
