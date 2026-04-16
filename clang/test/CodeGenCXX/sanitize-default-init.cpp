// RUN: %clang_cc1 -x c++ -fsanitize=memory -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

/// Sanitise the placement new with default initialisation style.

namespace std {
    using size_t = decltype(sizeof(0));
}

void *operator new(std::size_t, void *p) noexcept { return p; }

struct Simple {
    int x;
};

struct WithCtor {
    int x;
    int y[4];
    WithCtor() {
        bool flag = x > 0; /// This is UB
    }
};

// CHECK-LABEL: define {{.*}} i32 @main()
int main() {
    {
        Simple s;
        // CHECK: [[S:%.+]] = alloca %struct.Simple, align 4
        // CHECK: [[W:%.+]] = alloca %struct.WithCtor, align 4
        s.x = 42;
        // CHECK: call void @__msan_poison(ptr [[S]], i64 4)
        new (&s) Simple;
        bool flag = s.x == 42; /// This is UB
    }
    {
        WithCtor w;
        w.x = 42;
        // CHECK: call void @__msan_poison(ptr [[W]], i64 20)
        auto *ptr = new (&w) WithCtor; /// This is UB
        // CHECK: call void @_ZN8WithCtorC1Ev
    }
}
