// REQUIRES: x86-registered-target

//
// Check help message.
//
// RUN: clang-offload-wrapper --help | FileCheck %s --check-prefix CHECK-HELP
// CHECK-HELP: {{.*}}OVERVIEW: A tool to create a wrapper bitcode for offload target binaries.
// CHECK-HELP: {{.*}}Takes offload target binaries as input and produces bitcode file containing
// CHECK-HELP: {{.*}}target binaries packaged as data and initialization code which registers target
// CHECK-HELP: {{.*}}binaries in offload runtime.
// CHECK-HELP: {{.*}}USAGE: clang-offload-wrapper [options] <input  files>
// CHECK-HELP: {{.*}}-desc-name=<name>       - Specifies offload descriptor symbol name: '.<offload kind>.<name>', and makes it globally visible
// CHECK-HELP: {{.*}}-emit-entry-table       - Emit offload entry table
// CHECK-HELP: {{.*}}-emit-reg-funcs         - Emit [un-]registration functions
// CHECK-HELP: {{.*}}-o=<filename>           - Output filename
// CHECK-HELP: {{.*}}-reg-func-name=<name>   - Offload descriptor registration function name
// CHECK-HELP: {{.*}}-target=<kind-triple>   - Offload kind + target triple of the wrapper object: <offload kind>-<target triple>
// CHECK-HELP: {{.*}}-unreg-func-name=<name> - Offload descriptor un-registration function name

// -------
// Generate a file to wrap.
//
// RUN: echo 'Content of device file' > %t.tgt
//
// -------
// Check bitcode produced by the wrapper tool.
//
// RUN: clang-offload-wrapper -target=openmp-x86_64-pc-linux-gnu -o - %t.tgt | llvm-dis | FileCheck %s --check-prefix CHECK-IR

// CHECK-IR: target triple = "x86_64-pc-linux-gnu"

// CHECK-IR-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }
// CHECK-IR-DAG: [[IMAGETY:%.+]] = type { i8*, i8*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-IR-DAG: [[DESCTY:%.+]] = type { i32, [[IMAGETY]]*, [[ENTTY]]*, [[ENTTY]]* }

// CHECK-IR: [[ENTBEGIN:@.+]] = external constant [[ENTTY]]
// CHECK-IR: [[ENTEND:@.+]] = external constant [[ENTTY]]

// CHECK-IR: [[BIN:@.+]] = internal unnamed_addr constant [[BINTY:\[[0-9]+ x i8\]]] c"Content of device file{{.+}}"

// CHECK-IR: [[IMAGES:@.+]] = internal unnamed_addr constant [1 x [[IMAGETY]]] [{{.+}} { i8* getelementptr inbounds ([[BINTY]], [[BINTY]]* [[BIN]], i64 0, i64 0), i8* getelementptr inbounds ([[BINTY]], [[BINTY]]* [[BIN]], i64 1, i64 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }]

// CHECK-IR: [[DESC:@.+]] = internal constant [[DESCTY]] { i32 1, [[IMAGETY]]* getelementptr inbounds ([1 x [[IMAGETY]]], [1 x [[IMAGETY]]]* [[IMAGES]], i64 0, i64 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }

// CHECK-IR: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* [[REGFN:@.+]], i8* null }]
// CHECK-IR: @llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* [[UNREGFN:@.+]], i8* null }]

// CHECK-IR: define internal void [[REGFN]]()
// CHECK-IR:   call void @__tgt_register_lib([[DESCTY]]* [[DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_register_lib([[DESCTY]]*)

// CHECK-IR: define internal void [[UNREGFN]]()
// CHECK-IR:   call void @__tgt_unregister_lib([[DESCTY]]* [[DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_unregister_lib([[DESCTY]]*)

// -------
// Check options' effects: -emit-reg-funcs, -emit-entry-table=0, -desc-name
//
// RUN: clang-offload-wrapper -target=sycl-x86_64-pc-linux-gnu -emit-reg-funcs=0 -emit-entry-table=0 -desc-name=lalala -o - %t.tgt | llvm-dis | FileCheck %s --check-prefix CHECK-IR1
// CHECK-IR1: source_filename = "offload.wrapper.object"
// CHECK-IR1: [[IMAGETY:%.+]] = type { i8*, i8* }
// CHECK-IR1: [[DESCTY:%.+]] = type { i32, [[IMAGETY]]* }
// CHECK-IR1-NOT: @llvm.global_ctors
// CHECK-IR1-NOT: @llvm.global_dtors
// CHECK-IR1: @.sycl_offloading.lalala = constant [[DESCTY]] { i32 1, [[IMAGETY]]* getelementptr inbounds ([1 x [[IMAGETY]]], [1 x [[IMAGETY]]]* @.sycl_offloading.device_images, i64 0, i64 0) }

// -------
// Check options' effects: -reg-func-name, -unreg-func-name
//
// RUN: clang-offload-wrapper -target=sycl-x86_64-pc-linux-gnu -reg-func-name=__REGFUNC__ -unreg-func-name=__UNREGFUNC__ -o - %t.tgt | llvm-dis | FileCheck %s --check-prefix CHECK-IR2
// CHECK-IR2: source_filename = "offload.wrapper.object"
// CHECK-IR2: define internal void {{.+}}()
// CHECK-IR2:   call void @__REGFUNC__
// CHECK-IR2:   ret void

// CHECK-IR2: declare void @__REGFUNC__

// CHECK-IR2: define internal void {{.+}}()
// CHECK-IR2:   call void @__UNREGFUNC__
// CHECK-IR2:   ret void

// CHECK-IR2: declare void @__UNREGFUNC__


