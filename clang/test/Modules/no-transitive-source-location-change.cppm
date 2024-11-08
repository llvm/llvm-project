// Testing that adding a new line in a module interface unit won't cause the BMI
// of consuming module unit changes.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.v1.cppm -emit-reduced-module-interface -o %t/A.v1.pcm
//
// The BMI may not be the same since the source location differs.
// RUN: not diff %t/A.pcm %t/A.v1.pcm &> /dev/null
//
// The BMI of B shouldn't change since all the locations remain the same.
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-reduced-module-interface -fmodule-file=A=%t/A.pcm \
// RUN:     -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -emit-reduced-module-interface -fmodule-file=A=%t/A.v1.pcm \
// RUN:     -o %t/B.v1.pcm
// RUN: diff %t/B.v1.pcm %t/B.pcm  &> /dev/null
//
// The BMI of C may change since the locations for instantiations changes.
// RUN: %clang_cc1 -std=c++20 %t/C.cppm -emit-reduced-module-interface -fmodule-file=A=%t/A.pcm \
// RUN:     -o %t/C.pcm
// RUN: %clang_cc1 -std=c++20 %t/C.cppm -emit-reduced-module-interface -fmodule-file=A=%t/A.v1.pcm \
// RUN:     -o %t/C.v1.pcm
// RUN: not diff %t/C.v1.pcm %t/C.pcm  &> /dev/null

//--- A.cppm
export module A;
export template <class T>
struct C {
    T func() {
        return T(43);
    }
};
export int funcA() {
    return 43;
}

//--- A.v1.cppm
export module A;

export template <class T>
struct C {
    T func() {
        return T(43);
    }
};
export int funcA() {
    return 43;
}

//--- B.cppm
export module B;
import A;

export int funcB() {
    return funcA();
}

//--- C.cppm
export module C;
import A;
export inline void testD() {
    C<int> c;
    c.func();
}
