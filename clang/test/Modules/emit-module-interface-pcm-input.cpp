// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t

// Related to issue #132692
// Verify that clang_cc1 doesn't crash when trying to generate a module
// interface from an alreay precompiled module (`.pcm`).
// RUN: %clang_cc1 -std=c++20 -emit-module-interface a.cppm -o a.pcm
// RUN: not %clang_cc1 -std=c++20 -emit-module-interface a.pcm
// RUN: not %clang_cc1 -std=c++20 -emit-reduced-module-interface a.pcm

//--- a.cppm
export module A;
