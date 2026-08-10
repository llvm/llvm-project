// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUNX: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUNX:     -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUNX: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUNX:     %t/b.cpp -fmodule-file=a=%t/a.pcm -disable-llvm-passes \
// RUNX:     -emit-llvm -o - | FileCheck %t/b.cpp
// RUNX: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUNX:     %t/c.cpp -fmodule-file=a=%t/a.pcm -disable-llvm-passes \
// RUNX:     -emit-llvm -o - | FileCheck %t/c.cpp

// Test again with reduced BMI.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     -emit-reduced-module-interface %t/a.cppm -o %t/a.pcm
// RUNX: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUNX:     %t/b.cpp -fmodule-file=a=%t/a.pcm -disable-llvm-passes \
// RUNX:     -emit-llvm -o - | FileCheck %t/b.cpp
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/c.cpp -fmodule-file=a=%t/a.pcm -disable-llvm-passes \
// RUN:     -emit-llvm -o - | FileCheck %t/c.cpp

//--- a.cppm
export module a;

struct integer {
	explicit operator int() const {
		return 0;
	}
};

export template<typename>
int a = static_cast<int>(integer());

int aa() {
	return a<void>;
}


//--- b.cpp
import a;

void b() {}

// CHECK-NOT: @_ZW1a1dIvE =
// CHECK-NOT: @_ZGVW1a1dIvE =
// CHECK-NOT: @_ZW1a11dynamic_var =
// CHECK-NOT: @_ZGVW1a11dynamic_var =
// CHECK-NOT: @_ZW1a1aIvE =
// CHECK-NOT: @_ZGVW1a1aIvE =

//--- c.cpp
import a;
int c() {
    return a<void>;
}

// The used variables are generated normally
// CHECK-DAG: @_ZW1a1aIvE =
// CHECK-DAG: @_ZGVW1a1aIvE =
