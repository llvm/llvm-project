// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/impl.cppm -emit-reduced-module-interface -o %t/impl.pcm
// RUN: %clang_cc1 -std=c++20 %t/impl.v2.cppm -emit-reduced-module-interface -o %t/impl.v2.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm \
// RUN:     -fmodule-file=impl=%t/impl.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.v2.pcm \
// RUN:     -fmodule-file=impl=%t/impl.v2.pcm

// Since m only uses impl in the definition, the change in impl shouldn't affect m.
// RUN: diff %t/m.pcm %t/m.v2.pcm &> /dev/null

//--- impl.cppm
export module impl;
export struct Impl {
    Impl() {}
    Impl(const Impl &) {}
    ~Impl() {}
    int get() {
        return 43;
    }
};

//--- impl.v2.cppm
export module impl;
export struct Impl {
    Impl() {}
    Impl(const Impl &) {}
    ~Impl() {}
    int get() {
        return 43;
    }
};

export struct ImplV2 {
    int get() {
        return 43;
    }
};

//--- m.cppm
export module m;
import impl;

export int interface() {
    Impl impl;
    return impl.get();
};
