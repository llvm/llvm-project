// REQUIRES: system-windows, target={{.*-windows-msvc}}
// RUN: %clang_cc1 -x c++ -std=c++17 -fcxx-exceptions -fexceptions -triple=%ms_abi_triple -emit-pch -building-pch-with-obj -fmodules-codegen -o %t.pch %S/cxx-exception-copy-ctor-crash.h
// RUN: %clang_cc1 -x c++ -std=c++17 -fcxx-exceptions -fexceptions -triple=%ms_abi_triple -include-pch %t.pch -emit-obj -building-pch-with-obj -fmodules-codegen -o %t.pch.obj
// RUN: %clang_cc1 -x c++ -std=c++17 -fcxx-exceptions -fexceptions -triple=%ms_abi_triple -include-pch %t.pch -emit-obj -o %t.obj %s
// RUN: lld-link -subsystem:console -out:%t.exe %t.pch.obj %t.obj libucrt.lib libvcruntime.lib libcmt.lib
// RUN: %t.exe

// Regression test for https://github.com/llvm/llvm-project/issues/53486

int main() {
  try {
    throw Exception();
  } catch (const Exception ex) { // catch by value to trigger copy constructor
  }
  if (ctor_count != dtor_count) {
    return 1;
  }
  return 0;
}
