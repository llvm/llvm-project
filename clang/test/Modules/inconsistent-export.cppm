// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/m-a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/m-b.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-module-interface -o %t/m.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/use.cppm -fprebuilt-module-path=%t -emit-obj

// Test again with reduced BMI.
// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/m-a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -o %t/m-b.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/use.cppm -fprebuilt-module-path=%t -emit-obj


//--- a.cppm
export module m:a;
namespace n {
export class a {
public:
    virtual ~a() {}
};
}

//--- b.cppm
export module m:b;
namespace n {
class a;
}

//--- m.cppm
export module m;
export import :a;
export import :b;

//--- use.cppm
// expected-no-diagnostics
export module u;
export import m;

struct aa : public n::a {
    aa() {}
};
auto foo(n::a*) {
    return;
}

void use() {
    n::a _;
}
