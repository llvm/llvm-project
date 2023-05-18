// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pch -o %t1 %S/pchpch1.h
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pch -o %t2 %S/pchpch2.h -include-pch %t1
// RUN: %clang_cc1 -triple i386-unknown-unknown -fsyntax-only %s -include-pch %t2
// REQUIRES: x86-registered-target

// The purpose of this test is to make sure that a PCH created while including
// an existing PCH can be loaded.
