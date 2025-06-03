// RUN: %clang_cc1 -triple s390x-ibm-zos -fzos-extensions -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -fzos-extensions -fxl-pragma-pack -fsyntax-only -verify %s
// RUN: %clang -target s390x-ibm-zos -S -emit-llvm -Xclang -verify -fno-xl-pragma-pack -o %t.ll %s

#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
#pragma pack(twobyte)
#pragma pack(packed)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 1}}
#pragma pack(reset)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 2}}
#pragma pack(pop)
#pragma pack(show) // expected-warning {{value of #pragma pack(show) == 8}}
