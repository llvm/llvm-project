// REQUIRES: x86-registered-target

//
// Check help message.
//
// RUN: clang-offload-wrapper --help | FileCheck %s --check-prefix CHECK-HELP
// CHECK-HELP: {{.*}}OVERVIEW: A tool to create a wrapper bitcode for offload target binaries. Takes offload
// CHECK-HELP: {{.*}}target binaries as input and produces bitcode file containing target binaries packaged
// CHECK-HELP: {{.*}}as data and initialization code which registers target binaries in offload runtime.
// CHECK-HELP: {{.*}}USAGE: clang-offload-wrapper [options] <input files>
// CHECK-HELP: {{.*}}  -o=<filename>               - Output filename
// CHECK-HELP: {{.*}}  --target=<triple>           - Target triple for the output module

//
// Generate a file to wrap.
//
// RUN: echo 'Content of device file' > %t.tgt

//
// Check bitcode produced by the wrapper tool.
//
// RUN: clang-offload-wrapper -target=x86_64-pc-linux-gnu -o %t.wrapper.bc %t.tgt
// RUN: llvm-dis %t.wrapper.bc -o - | FileCheck %s --check-prefix CHECK-IR

// CHECK-IR: target triple = "x86_64-pc-linux-gnu"

// CHECK-IR-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }
// CHECK-IR-DAG: [[IMAGETY:%.+]] = type { i8*, i8*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-IR-DAG: [[DESCTY:%.+]] = type { i32, [[IMAGETY]]*, [[ENTTY]]*, [[ENTTY]]* }

// CHECK-IR: [[ENTBEGIN:@.+]] = external hidden constant [[ENTTY]]
// CHECK-IR: [[ENTEND:@.+]] = external hidden constant [[ENTTY]]

// CHECK-IR: [[DUMMY:@.+]] = hidden constant [0 x [[ENTTY]]] zeroinitializer, section "omp_offloading_entries"

// CHECK-IR: [[BIN:@.+]] = internal unnamed_addr constant [[BINTY:\[[0-9]+ x i8\]]] c"Content of device file{{.+}}"

// CHECK-IR: [[IMAGES:@.+]] = internal unnamed_addr constant [1 x [[IMAGETY]]] [{{.+}} { i8* getelementptr inbounds ([[BINTY]], [[BINTY]]* [[BIN]], i64 0, i64 0), i8* getelementptr inbounds ([[BINTY]], [[BINTY]]* [[BIN]], i64 1, i64 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }]

// CHECK-IR: [[DESC:@.+]] = internal constant [[DESCTY]] { i32 1, [[IMAGETY]]* getelementptr inbounds ([1 x [[IMAGETY]]], [1 x [[IMAGETY]]]* [[IMAGES]], i64 0, i64 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }

// CHECK-IR: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* [[REGFN:@.+]], i8* null }]
// CHECK-IR: @llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* [[UNREGFN:@.+]], i8* null }]

// CHECK-IR: define internal void [[REGFN]]()
// CHECK-IR:   call void @__tgt_register_lib([[DESCTY]]* [[DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_register_lib([[DESCTY]]*)

// CHECK-IR: define internal void [[UNREGFN]]()
// CHECK-IR:   call void @__tgt_unregister_lib([[DESCTY]]* [[DESC]])
// CHECK-IR:   ret void

// CHECK-IR: declare void @__tgt_unregister_lib([[DESCTY]]*)
