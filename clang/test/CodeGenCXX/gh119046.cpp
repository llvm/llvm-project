// RUN: %clang_cc1 -std=c++2a -triple x86_64-elf-gnu %s -emit-llvm -o - | FileCheck %s

struct S {
    consteval void operator()() {}
};

template <class Fn>
constexpr void dispatch(Fn fn) {
    fn();
}

template <class Visitor>
struct value_visitor {
    constexpr void operator()() { visitor(); }
    Visitor&& visitor;
};

template <class Visitor>
constexpr auto make_dispatch() {
    return dispatch<value_visitor<S>>;
}

template <class Visitor>
constexpr void visit(Visitor&&) {
    make_dispatch<Visitor>();
}

void f() { visit(S{}); }

// CHECK: define {{.*}} @_Z1fv
// CHECK-NOT: define {{.*}} @_Z5visitI1SEvOT_
// CHECK-NOT: define {{.*}} @_Z13make_dispatchI1SEDav
