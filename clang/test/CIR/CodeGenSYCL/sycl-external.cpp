// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -fsycl-is-device -triple spirv64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Verify that sycl_external functions are emitted in device code, matching
// classic CodeGen, and that their definitions carry the "sycl-module-id"
// attribute that marks them as entry points for device-code splitting.

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct KN;
void use() { kernel_single_task<KN>([] {}); }

// Defined and not used: emitted.
[[clang::sycl_external]] int square(int x) { return x * x; }

// Declared but not defined or used: not emitted.
[[clang::sycl_external]] int declOnly();

// Declared and used in device code but not defined: external reference.
[[clang::sycl_external]] void declUsedInDevice(int y);
[[clang::sycl_external]] void deviceUse() { declUsedInDevice(3); }

// Declared with the attribute and later defined: definition emitted.
[[clang::sycl_external]] int func1(int arg);
int func1(int arg) { return arg; }

// Reachable from a sycl_external function: emitted, but not an entry point.
int ret1() { return 1; }
[[clang::sycl_external]] int withAttr() { return ret1(); }

// Explicit specialization defined: emitted.
template <typename T> [[clang::sycl_external]] void tFunc(T arg) {}
template <> [[clang::sycl_external]] void tFunc<int>(int arg) {}

// Defined without the attribute and not used in device code: not emitted.
int squareNoAttr(int x) { return x * x; }

// CIR: cir.func {{.*}}@_Z6squarei({{.*}} attributes {{{.*}}"sycl-module-id" = "{{.*}}sycl-external.cpp"}
// CIR: cir.func {{.*}}@_Z9deviceUsev() {{.*}} attributes {{{.*}}"sycl-module-id" = "{{.*}}sycl-external.cpp"}
// CIR: cir.func private @_Z16declUsedInDevicei({{.*}} attributes {convergent{{[^"]*}}} loc
// CIR: cir.func {{.*}}@_Z5func1i({{.*}} attributes {{{.*}}"sycl-module-id" = "{{.*}}sycl-external.cpp"}
// CIR: cir.func {{.*}}@_Z8withAttrv() {{.*}} attributes {{{.*}}"sycl-module-id" = "{{.*}}sycl-external.cpp"}
// CIR: cir.func {{.*}}@_Z4ret1v() {{.*}} attributes {convergent{{[^"]*}}} {
// CIR: cir.func {{.*}}@_Z5tFuncIiEvT_({{.*}} attributes {{{.*}}"sycl-module-id" = "{{.*}}sycl-external.cpp"}
// CIR: cir.func {{.*}}@_ZTS2KN({{.*}} attributes {{{.*}}"sycl-module-id" = "{{.*}}sycl-external.cpp"}
// CIR-NOT: @_Z8declOnlyv
// CIR-NOT: @_Z12squareNoAttri

// LLVM: define spir_func noundef i32 @_Z6squarei(i32 noundef %{{.*}}) #[[EXT:[0-9]+]]
// LLVM: define spir_func void @_Z9deviceUsev() #[[EXT]]
// LLVM: declare spir_func void @_Z16declUsedInDevicei(i32 noundef) #[[DECL:[0-9]+]]
// LLVM: define spir_func noundef i32 @_Z5func1i(i32 noundef %{{.*}}) #[[EXT]]
// LLVM: define spir_func noundef i32 @_Z8withAttrv() #[[EXT]]
// LLVM: define spir_func noundef i32 @_Z4ret1v() #[[NOEXT:[0-9]+]]
// LLVM: define spir_func void @_Z5tFuncIiEvT_(i32 noundef %{{.*}}) #[[EXT]]
// ClangIR does not yet pass kernel arguments byval or emit the attributes that
// would let the kernel share the attribute group of the other entry points.
// LLVM-CIR: define spir_kernel void @_ZTS2KN({{.*}}) #[[KERNEL:[0-9]+]]
// OGCG: define spir_kernel void @_ZTS2KN(ptr noundef byval({{.*}}) #[[EXT]]
// LLVM-NOT: @_Z8declOnlyv
// LLVM-NOT: @_Z12squareNoAttri
// LLVM-DAG: attributes #[[EXT]] = { {{.*}}"sycl-module-id"="{{.*}}sycl-external.cpp" }
// LLVM-DAG: attributes #[[DECL]] = { convergent{{.*}} }
// LLVM-DAG: attributes #[[NOEXT]] = { convergent {{.*}}noinline{{.*}} }
// LLVM-CIR-DAG: attributes #[[KERNEL]] = { {{.*}}"sycl-module-id"="{{.*}}sycl-external.cpp" }
