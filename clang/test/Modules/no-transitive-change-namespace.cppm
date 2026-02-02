// Test that adding a new decl in a module which originally only contained a namespace
// won't break the dependency.
//
// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/Root.cppm -o %t/Root.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/N.cppm -o %t/N.pcm \
// RUN:     -fmodule-file=Root=%t/Root.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/N.v1.cppm -o %t/N.v1.pcm \
// RUN:     -fmodule-file=Root=%t/Root.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/M.cppm -o %t/M.pcm \
// RUN:     -fmodule-file=N=%t/N.pcm -fmodule-file=Root=%t/Root.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/M.cppm -o %t/M.v1.pcm \
// RUN:     -fmodule-file=N=%t/N.v1.pcm -fmodule-file=Root=%t/Root.pcm
//
// RUN: diff %t/M.pcm %t/M.v1.pcm &> /dev/null

//--- Root.cppm
export module Root;
export namespace N {

}

//--- N.cppm
export module N;
import Root;
export namespace N {

}

//--- N.v1.cppm
export module N;
import Root;
export namespace N {
struct NN {};
}

//--- M.cppm
export module M;
import N;
export namespace N {
struct MM {};
}
