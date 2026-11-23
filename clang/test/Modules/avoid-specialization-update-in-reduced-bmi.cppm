// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/base.cppm -emit-module-interface -o %t/base.pcm
// RUN: %clang_cc1 -std=c++20 %t/update.cppm -fmodule-file=base=%t/base.pcm -emit-module-interface -o %t/update.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/update.pcm | FileCheck %t/update.cppm --check-prefix=FULL
//
// RUN: %clang_cc1 -std=c++20 %t/base.cppm -emit-reduced-module-interface -o %t/base.pcm
// RUN: %clang_cc1 -std=c++20 %t/update.cppm -fmodule-file=base=%t/base.pcm -emit-reduced-module-interface -o %t/update.pcm
// RUN: llvm-bcanalyzer --dump --disable-histogram %t/update.pcm | FileCheck %t/update.cppm

//--- base.cppm
export module base;

export template <typename T>
struct base {
    T value;
};

//--- update.cppm
export module update;
import base;
export int update() {
    return base<int>().value;
}

// FULL: TEMPLATE_SPECIALIZATION
// CHECK-NOT: TEMPLATE_SPECIALIZATION
