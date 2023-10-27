// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple \
// RUN:     %t/b.cpp -fmodule-file=a=%t/a.pcm -disable-llvm-passes \
// RUN:     -emit-llvm -o - | FileCheck %t/b.cpp
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

struct s {
    ~s();
    operator int() const;
};

export template<typename>
auto d = s();

int aa() {
	return a<void> + d<void>;
}

int dynamic_func();
export inline int dynamic_var = dynamic_func();

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
    return a<void> + d<void> + dynamic_var;
}

// The used variables are generated normally
// CHECK-DAG: @_ZW1a1aIvE =
// CHECK-DAG: @_ZW1a1dIvE =
// CHECK-DAG: @_ZW1a11dynamic_var = linkonce_odr
// CHECK-DAG: @_ZGVW1a1aIvE =
// CHECk-DAG: @_ZGVW1a1dIvE =
// CHECK-DAG: @_ZGVW1a11dynamic_var = linkonce_odr
