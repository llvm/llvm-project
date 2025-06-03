// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:   -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++23 -fprebuilt-module-path=%t %t/b.pcm -emit-llvm \
// RUN:     -disable-llvm-passes -o - | FileCheck %t/b.cppm

//--- a.cppm
module;

struct base {
    virtual void f() const;
};

inline void base::f() const {
}

export module a;
export using ::base;

//--- b.cppm
module;

struct base {
    virtual void f() const;
};

inline void base::f() const {
}

export module b;
import a;
export using ::base;

export extern "C" void func() {}

// We only need to check that the IR are successfully emitted instead of crash.
// CHECK: func
