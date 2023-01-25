// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-pch -o %t.pch
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -emit-llvm -include-pch %t.pch -o - | FileCheck %s

#ifndef HEADER
#define HEADER

class OOArray{
public:
  ~OOArray();
};

class OOString {
public:
    OOString();
    OOString(char *);
};

class OOPattern {
public:
    OOArray matchAll(const OOString &)const {
        __attribute__((__blocks__(byref))) OOArray out;
    }
};

OOArray operator & (const OOPattern & pattern) {
    pattern.matchAll(0);
}
OOArray operator & (OOString, OOString);

#else

// We just make sure there is no crash on IRGen (rdar://13114142)
// CHECK: _Z3foov()
void foo() {
  OOString str;
  str & "o";
}

#endif
