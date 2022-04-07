// RUN: %clang_cc1 -no-opaque-pointers -triple amdgcn-amd-amdhsa -emit-llvm %s -o - | FileCheck %s

// CHECK: @.str = private unnamed_addr addrspace(4) constant [6 x i8] c"g_str\00", align 1
// CHECK: @g_str ={{.*}} addrspace(1) global i8* addrspacecast (i8 addrspace(4)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(4)* @.str, i32 0, i32 0) to i8*), align 8
// CHECK: @g_array ={{.*}} addrspace(1) global [8 x i8] c"g_array\00", align 1
// CHECK: @.str.1 = private unnamed_addr addrspace(4) constant [6 x i8] c"l_str\00", align 1
// CHECK: @__const._Z1fv.l_array = private unnamed_addr addrspace(4) constant [8 x i8] c"l_array\00", align 1

const char* g_str = "g_str";
char g_array[] = "g_array";

void g(const char* p);

// CHECK-LABEL: define{{.*}} void @_Z1fv()
void f() {
  const char* l_str = "l_str";

  // CHECK: call void @llvm.memcpy.p0i8.p4i8.i64
  char l_array[] = "l_array";

  g(g_str);
  g(g_array);
  g(l_str);
  g(l_array);

  const char* p = g_str;
  g(p);
}

// CHECK-LABEL: define{{.*}} void @_Z1ev
void e() {
  g("string literal");
  g("string literal");
}
