// Test that adding a new unused decl within reduced BMI may not produce a transitive change.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/T.cppm -o %t/T.pcm   \
// RUN:     -fmodule-file=T=%t/T.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/T1.cppm -o %t/T1.pcm \
// RUN:     -fmodule-file=T=%t/T.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/T2.cppm -o %t/T2.pcm \
// RUN:     -fmodule-file=T=%t/T.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/T3.cppm -o %t/T3.pcm \
// RUN:     -fmodule-file=T=%t/T.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/T4.cppm -o %t/T4.pcm \
// RUN:     -fmodule-file=T=%t/T.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/A.cppm -o %t/A.pcm \
// RUN:     -fmodule-file=T=%t/T.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/AWrapper.cppm -o %t/AWrapper.pcm \
// RUN:      -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/B.cppm -o %t/B.pcm \
// RUN:     -fprebuilt-module-path=%t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/A.v1.cppm -o %t/A.v1.pcm \
// RUN:     -fmodule-file=T=%t/T.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/AWrapper.cppm -o %t/AWrapper.v1.pcm \
// RUN:      -fprebuilt-module-path=%t -fmodule-file=A=%t/A.v1.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/B.cppm -o %t/B.v1.pcm \
// RUN:     -fprebuilt-module-path=%t -fmodule-file=AWrapper=%t/AWrapper.v1.pcm -fmodule-file=A=%t/A.v1.pcm
//
// RUN: not diff %t/B.pcm %t/B.v1.pcm &> /dev/null

//--- T.cppm
export module T;
export template <class T>
struct Templ {};

//--- T1.cppm
export module T1;
import T;
export using Tunsigned = Templ<unsigned>;

//--- T2.cppm
export module T2;
import T;
export using Tfloat = Templ<float>;

//--- T3.cppm
export module T3;
import T;
export using Tlong = Templ<long long>;

//--- T4.cppm
export module T4;
import T;
export using Tshort = Templ<short>;

//--- A.cppm
export module A;
import T;
export using Tint = Templ<int>;

//--- A.v1.cppm
export module A;
import T;
export using Tint = Templ<int>;
void __unused__() {}

//--- AWrapper.cppm
export module AWrapper;
import A;

//--- B.cppm
export module B;
import AWrapper;
import T;
import T1;
import T2;
import T3;
import T4;

export using Tdouble = Templ<double>;
