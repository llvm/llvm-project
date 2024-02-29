// From https://github.com/llvm/llvm-project/issues/61067
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm \
// RUN:     -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/b.cppm \
// RUN:     -emit-module-interface -fmodule-file=a=%t/a.pcm -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/b.pcm -S \
// RUN:     -emit-llvm -fmodule-file=a=%t/a.pcm -disable-llvm-passes -o - | FileCheck %t/b.cppm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/c.cpp -fmodule-file=a=%t/a.pcm \
// RUN:     -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/c.cpp

//--- a.cppm
export module a;

export struct a {
	friend bool operator==(a, a) = default;
};

//--- b.cppm
export module b;

import a;

void b() {
	(void)(a() == a());
}

// CHECK: define{{.*}}linkonce_odr{{.*}}@_ZW1aeqS_1aS0_(

//--- c.cpp
import a;

int c() {
    (void)(a() == a());
}

// CHECK: define{{.*}}linkonce_odr{{.*}}@_ZW1aeqS_1aS0_(
