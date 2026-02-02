// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -dwarf-version=4 -debug-info-kind=constructor \
// RUN:     -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -dwarf-version=4 -debug-info-kind=constructor \
// RUN:     -emit-module-interface -o %t/b.pcm -fmodule-file=a=%t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -dwarf-version=4 -debug-info-kind=constructor \
// RUN:     -emit-llvm -o - -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm | FileCheck %t/b.cpp 

//--- a.cppm
export module a;
export template <class T>
class a {
private:
    T *data;

public:
    virtual T* getData();
};

extern template class a<char>;

//--- b.cppm
export module b;
import a;
export struct b {
    a<char> v;
};

//--- b.cpp
module b;
extern "C" void func() {
    b();
}

// It is fine enough to check that we won't crash.
// CHECK: define {{.*}}void @func()
