// RUN: %clang_cc1 \
// RUN:     -triple x86_64-scei-ps4 \
// RUN:     -fdeclspec \
// RUN:     -Werror \
// RUN:     -emit-llvm %s -o - | \
// RUN:   FileCheck %s

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

// CHECK-DAG: @import_int = external dllimport
// CHECK-DAG: @export_int ={{.*}} dllexport global i32 0
// CHECK-DAG: define{{.*}} dllexport void @export_implemented_function()
// CHECK-DAG: declare dllimport void @import_function(i32)
