// Address: https://github.com/llvm/llvm-project/issues/60693
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -fmodule-file=a=%t/a.pcm %t/c.cpp -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/c.cpp

// Test again with reduced BMI
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -triple %itanium_abi_triple -fmodule-file=a=%t/a.pcm %t/c.cpp -S -emit-llvm -disable-llvm-passes -o - | FileCheck %t/c.cpp

//--- a.cppm
export module a;

constexpr bool f() {
	for (unsigned n = 0; n != 10; ++n) {
	}
	return true;
}

template <typename T>
struct u {
    T unit() {
        return T();
    }
};

export template<typename T>
struct s {
	static constexpr auto a = f();
	static constexpr auto b = f();
	static constexpr auto c = f();
	static constexpr auto d = f();
    int foo() {
        return 43;
    }
    int bar() {
        return 44;
    }
    T zoo() {
        return u<T>().unit();
    }
};

template struct s<int>;
template struct s<long>;

//--- c.cpp
import a;

extern "C" int use() {
    s<int> _;
    return _.a + _.b + _.c + _.d;
}

extern "C" long use2() {
    s<long> _;
    return _.foo();
}

extern "C" long use3() {
    s<long> _;
    return _.zoo();
}

// CHECK: define{{.*}}@use(
// CHECK-NOT: }
// CHECK: ret{{.*}} 4

// CHECK: declare{{.*}}@_ZNW1a1sIlE3fooEv
// CHECK-NOT: _ZNW1a1sIlE3barEv
// CHECK: declare{{.*}}_ZNW1a1sIlE3zooEv
