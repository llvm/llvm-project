// RUN: c-index-test -test-load-source all -x cuda %s | FileCheck %s
// RUN: c-index-test -test-load-source all -x cuda --cuda-host-only %s | FileCheck %s
// RUN: c-index-test -test-load-source all -x cuda --cuda-device-only %s | FileCheck %s

__attribute__((device)) void f_device();
__attribute__((global)) void f_global();
__attribute__((constant)) int* g_constant;
__attribute__((shared)) float *g_shared;
__attribute__((host)) void f_host();
__attribute__((device_builtin)) void f_device_builtin();
typedef __attribute__((device_builtin)) const void *t_device_builtin;
enum __attribute__((device_builtin)) e_device_builtin {};
__attribute__((device_builtin)) int v_device_builtin;
__attribute__((cudart_builtin)) void f_cudart_builtin();
__attribute__((nv_weak)) void f_nv_weak();
__attribute__((device_builtin_surface_type)) unsigned long long surface_var;
__attribute__((device_builtin_texture_type)) unsigned long long texture_var;

// CHECK:       attributes-cuda.cu:5:30: FunctionDecl=f_device:5:30
// CHECK-NEXT:  attributes-cuda.cu:5:16: attribute(device)
// CHECK:       attributes-cuda.cu:6:30: FunctionDecl=f_global:6:30
// CHECK-NEXT:  attributes-cuda.cu:6:16: attribute(global)
// CHECK:       attributes-cuda.cu:7:32: VarDecl=g_constant:7:32 (Definition)
// CHECK-NEXT:  attributes-cuda.cu:7:16: attribute(constant)
// CHECK:       attributes-cuda.cu:8:32: VarDecl=g_shared:8:32 (Definition)
// CHECK-NEXT:  attributes-cuda.cu:8:16: attribute(shared)
// CHECK:       attributes-cuda.cu:9:28: FunctionDecl=f_host:9:28
// CHECK-NEXT:  attributes-cuda.cu:9:16: attribute(host)
// CHECK:	attributes-cuda.cu:10:38: FunctionDecl=f_device_builtin:10:38
// CHECK:	attributes-cuda.cu:11:53: TypedefDecl=t_device_builtin:11:53
// CHECK:	attributes-cuda.cu:12:38: EnumDecl=e_device_builtin:12:38
// CHECK:	attributes-cuda.cu:13:37: VarDecl=v_device_builtin:13:37
// CHECK:	attributes-cuda.cu:14:38: FunctionDecl=f_cudart_builtin:14:38
// CHECK:	attributes-cuda.cu:15:31: FunctionDecl=f_nv_weak:15:31
// CHECK:	attributes-cuda.cu:16:65: VarDecl=surface_var:16:65
// CHECK:	attributes-cuda.cu:17:65: VarDecl=texture_var:17:65
