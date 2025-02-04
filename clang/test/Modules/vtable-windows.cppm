// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple i686-pc-windows-msvc %t/foo.cppm -emit-module-interface \
// RUN:   -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -triple i686-pc-windows-msvc %t/user.cc -fmodule-file=foo=%t/foo.pcm \
// RUN:   -emit-llvm -o - -disable-llvm-passes | FileCheck %t/user.cc

//--- foo.cppm
export module foo;
export struct Fruit {
    virtual ~Fruit() = default;
    virtual void eval();
};

//--- user.cc
import foo;
void test() {
  Fruit *f = new Fruit();
  f->eval();
}

// Check that the virtual table is an unnamed_addr constant in comdat that can
// be merged with the virtual table with other TUs.
// CHECK: unnamed_addr constant {{.*}}[ptr @"??_R4Fruit@@6B@", ptr @"??_GFruit@@UAEPAXI@Z", ptr @"?eval@Fruit@@UAEXXZ"{{.*}}comdat($"??_7Fruit@@6B@")
