// Testing that adding an new identifier in an unused module file won't change 
// the BMI of the current module file.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/m-partA.cppm -emit-reduced-module-interface -o %t/m-partA.pcm
// RUN: %clang_cc1 -std=c++20 %t/m-partA.v1.cppm -emit-reduced-module-interface -o \
// RUN:     %t/m-partA.v1.pcm
// RUN: %clang_cc1 -std=c++20 %t/m-partB.cppm -emit-reduced-module-interface -o %t/m-partB.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.pcm \
// RUN:     -fmodule-file=m:partA=%t/m-partA.pcm -fmodule-file=m:partB=%t/m-partB.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -o %t/m.v1.pcm \
// RUN:     -fmodule-file=m:partA=%t/m-partA.v1.pcm -fmodule-file=m:partB=%t/m-partB.pcm
//
// RUN: %clang_cc1 -std=c++20 %t/useBOnly.cppm -emit-reduced-module-interface -o %t/useBOnly.pcm \
// RUN:     -fmodule-file=m=%t/m.pcm -fmodule-file=m:partA=%t/m-partA.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm
// RUN: %clang_cc1 -std=c++20 %t/useBOnly.cppm -emit-reduced-module-interface -o %t/useBOnly.v1.pcm \
// RUN:     -fmodule-file=m=%t/m.v1.pcm -fmodule-file=m:partA=%t/m-partA.v1.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm
// Since useBOnly only uses partB from module M, the change in partA shouldn't affect
// useBOnly.
// RUN: diff %t/useBOnly.pcm %t/useBOnly.v1.pcm &> /dev/null
//
// RUN: %clang_cc1 -std=c++20 %t/useAOnly.cppm -emit-reduced-module-interface -o %t/useAOnly.pcm \
// RUN:     -fmodule-file=m=%t/m.pcm -fmodule-file=m:partA=%t/m-partA.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm
// RUN: %clang_cc1 -std=c++20 %t/useAOnly.cppm -emit-reduced-module-interface -o %t/useAOnly.v1.pcm \
// RUN:     -fmodule-file=m=%t/m.v1.pcm -fmodule-file=m:partA=%t/m-partA.v1.pcm \
// RUN:     -fmodule-file=m:partB=%t/m-partB.pcm
// useAOnly should differ
// RUN: not diff %t/useAOnly.pcm %t/useAOnly.v1.pcm &> /dev/null

//--- m-partA.cppm
export module m:partA;

export inline int getA() {
    return 43;
}

export class A {
public:
    int getMem();
};

export template <typename T>
class ATempl {
public:
    T getT();
};

//--- m-partA.v1.cppm
export module m:partA;

export inline int getA() {
    return 43;
}

// The consuming module which didn't use m:partA completely is expected to be
// not changed.
export inline int getA2() {
    return 88;
}

export class A {
public:
    int getMem();

    // The consuming module which didn't use m:partA completely is expected to be
    // not changed.
    int getMem2();
};

export template <typename T>
class ATempl {
public:
    T getT();
    T getT2();
};

//--- m-partB.cppm
export module m:partB;

export inline int getB() {
    return 430;
}

//--- m.cppm
export module m;
export import :partA;
export import :partB;

//--- useBOnly.cppm
export module useBOnly;
import m;

export inline int get() {
    return getB();
}

//--- useAOnly.cppm
export module useAOnly;
import m;

export inline int get() {
    return getA();
}
