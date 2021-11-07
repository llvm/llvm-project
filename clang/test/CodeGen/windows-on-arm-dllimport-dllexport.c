// RUN: %clang_cc1 -Werror -triple thumbv7-windows-itanium -mfloat-abi hard -fms-extensions -emit-llvm %s -o - | FileCheck %s

__declspec(dllexport) int export_int;

__declspec(dllimport) int import_int;

__declspec(dllexport) void export_declared_function();

__declspec(dllexport) void export_implemented_function() {
}

__declspec(dllimport) void import_function(int);

void call_imported_function() {
  export_declared_function();
  return import_function(import_int);
}

// CHECK: @import_int = external dllimport global i32
// CHECK: @export_int = dso_local dllexport global i32 0, align 4

// CHECK: define dso_local dllexport arm_aapcs_vfpcc void @export_implemented_function()

// CHECK: declare dllimport arm_aapcs_vfpcc void @import_function(i32 noundef)

