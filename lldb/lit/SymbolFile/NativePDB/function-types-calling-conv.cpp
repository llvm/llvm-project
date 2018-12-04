// clang-format off
// REQUIRES: lld

// RUN: %build --compiler=clang-cl --arch=32 --nodefaultlib -o %t.exe -- %s 
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/function-types-calling-conv.lldbinit | FileCheck %s


void __stdcall StdcallFn() {}
void __fastcall FastcallFn() {}
void __thiscall ThiscallFn() {}
void __cdecl CdeclFn() {}
void __vectorcall VectorcallFn() {}

auto sfn = &StdcallFn;
// CHECK: (void (*)() __attribute__((stdcall))) sfn = {{.*}}

auto ffn = &FastcallFn;
// CHECK: (void (*)() __attribute__((fastcall))) ffn = {{.*}}

auto tfn = &ThiscallFn;
// CHECK: (void (*)() __attribute__((thiscall))) tfn = {{.*}}

auto cfn = &CdeclFn;
// CHECK: (void (*)()) cfn = {{.*}}

auto vfn = &VectorcallFn;
// CHECK: (void (*)() __attribute__((vectorcall))) vfn = {{.*}}

int main(int argc, char **argv) {
  return 0;
}
