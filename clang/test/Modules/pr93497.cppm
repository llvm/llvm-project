// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/mod.cppm \
// RUN:     -emit-module-interface -o %t/mod.pcm
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/use.cpp \
// RUN:     -fmodule-file=mod=%t/mod.pcm -emit-llvm \
// RUN:     -o - | opt -S --passes=simplifycfg | FileCheck %t/use.cpp

//--- mod.cppm
export module mod;

export struct Thing {
    static const Thing One;
    explicit Thing(int raw) :raw(raw) { }
    int raw;
};

const Thing Thing::One = Thing(1);

export struct C {
    int value;
};
export const C ConstantValue = {1};

export const C *ConstantPtr = &ConstantValue;

C NonConstantValue = {1};
export const C &ConstantRef = NonConstantValue;

export struct NonConstexprDtor {
    constexpr NonConstexprDtor(int raw) : raw(raw) {}
    ~NonConstexprDtor();

    int raw;
};

export const NonConstexprDtor NonConstexprDtorValue = {1};

//--- use.cpp
import mod;

int consume(int);
int consumeC(C);

extern "C" __attribute__((noinline)) inline int unneeded() {
    return consume(43);
}

extern "C" __attribute__((noinline)) inline int needed() {
    return consume(43);
}

int use() {
    Thing t1 = Thing::One;
    return consume(t1.raw);
}

int use2() {
    if (ConstantValue.value)
        return consumeC(ConstantValue);
    return unneeded();
}

int use3() {
    auto Ptr = ConstantPtr;
    if (Ptr->value)
        return consumeC(*Ptr);
    return needed();
}

int use4() {
    auto Ref = ConstantRef;
    if (Ref.value)
        return consumeC(Ref);
    return needed();
}

int use5() {
    NonConstexprDtor V = NonConstexprDtorValue;
    if (V.raw)
        return consume(V.raw);
    return needed();
}

// CHECK: @_ZNW3mod5Thing3OneE = external
// CHECK: @_ZW3mod13ConstantValue ={{.*}}available_externally{{.*}} constant 
// CHECK: @_ZW3mod11ConstantPtr = external
// CHECK: @_ZW3mod16NonConstantValue = external
// CHECK: @_ZW3mod21NonConstexprDtorValue = external

// Check that the middle end can optimize the program by the constant information.
// CHECK-NOT: @unneeded(

// Check that the use of ConstantPtr won't get optimized incorrectly.
// CHECK-LABEL: @_Z4use3v(
// CHECK: @needed(

// Check that the use of ConstantRef won't get optimized incorrectly.
// CHECK-LABEL: @_Z4use4v(
// CHECK: @needed(

// Check that the use of NonConstexprDtorValue won't get optimized incorrectly.
// CHECK-LABEL: @_Z4use5v(
// CHECK: @needed(
