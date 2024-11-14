// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Type.cppm -emit-module-interface -o \
// RUN:     %t/MyVec-Type.pcm -triple=x86_64-linux-gnu 
// RUN:%clang_cc1 -std=c++20 %t/Vec.cppm -emit-module-interface -o \
// RUN:     %t/MyVec-Vec.pcm -fmodule-file=MyVec:Type=%t/MyVec-Type.pcm \
// RUN:     -triple=x86_64-linux-gnu 
// RUN: %clang_cc1 -std=c++20 %t/Vec2.cppm -emit-module-interface -o \
// RUN:     %t/MyVec-Vec2.pcm -fmodule-file=MyVec:Type=%t/MyVec-Type.pcm \
// RUN:     -triple=x86_64-linux-gnu 
// RUN: %clang_cc1 -std=c++20 %t/Calculator.cppm -emit-module-interface -o \
// RUN:     %t/MyVec-Calculator.pcm -fmodule-file=MyVec:Vec=%t/MyVec-Vec.pcm   \
// RUN:     -fmodule-file=MyVec:Vec2=%t/MyVec-Vec2.pcm \
// RUN:     -fmodule-file=MyVec:Type=%t/MyVec-Type.pcm \
// RUN:     -triple=x86_64-linux-gnu 
// RUN: %clang_cc1 -std=c++20 %t/MyVec-Calculator.pcm -emit-llvm \
// RUN:     -fmodule-file=MyVec:Vec=%t/MyVec-Vec.pcm   \
// RUN:     -fmodule-file=MyVec:Vec2=%t/MyVec-Vec2.pcm \
// RUN:     -fmodule-file=MyVec:Type=%t/MyVec-Type.pcm \
// RUN:     -triple=x86_64-linux-gnu -o - \
// RUN:     | FileCheck %t/Calculator.cppm

//--- Type.cppm
export module MyVec:Type;

template <class T> struct Size {
  auto total() const { return 1; }
};

//--- Vec.cppm
export module MyVec:Vec;
import :Type;

int size_ = Size<int>().total();

//--- Vec2.cppm
export module MyVec:Vec2;
import :Type;

struct Vec2 {
    Size<int> size_;
};

//--- Calculator.cppm
export module MyVec:Calculator;

import :Vec;
import :Vec2;

auto Calculate() { return Size<int>().total(); };

// Check the emitted module initializer to make sure we generate the module unit
// successfully.
// CHECK: @_ZW5MyVec9Calculatev
