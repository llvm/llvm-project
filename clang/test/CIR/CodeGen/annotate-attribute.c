// RUN: %clang_cc1 -triple aarch64-linux-android29 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-linux-android29 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-linux-android29 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Verify that __attribute__((annotate(...))) is properly handled in CIR.
// The annotations are preserved as cir.annotation attributes on globals and
// functions. This is used by the Android NDK (__INTRODUCED_IN macro).
//
// LLVM lowering emits @llvm.global.annotations matching OGCG (with one
// entry per annotation per annotated value). String constants and per-arg
// structs are deduplicated by content / ArrayAttr identity respectively,
// so two functions with identical args share an .args global.

// Globals are emitted before functions in CIR, so check them first.

__attribute__((annotate("introduced_in=29")))
int annotated_var = 42;

// All globals (including static locals) are emitted before functions in CIR;
// check them all together up here so subsequent function CHECKs can rely on
// strict-order matching for the function section.
// CIR-DAG: cir.global external @annotated_var = #cir.int<42> : !s32i [#cir.annotation<"introduced_in=29">]
// CIR-DAG: cir.global "private" internal{{.*}} @{{.*counter.*}} = #cir.int<0> : !s32i [#cir.annotation<"static_local">]
// LLVM-DAG: @annotated_var = global i32 42
// LLVM-DAG: @{{.*counter.*}} = internal{{.*}} global i32 0
// OGCG-DAG: @annotated_var = global i32 42
// OGCG-DAG: @{{.*counter.*}} = internal{{.*}} global i32 0
// OGCG-DAG: @llvm.global.annotations = appending global

// CIR-side LLVM lowering also emits @llvm.global.annotations and the
// supporting strings/args constants. The CHECKs below cover each branch
// of the lowering helpers (string cache hit + miss, isArg=true vs false,
// args cache hit + miss, named-with-index vs named-without-index).
//
// (a) The first-emitted annotation-name string uses the no-index name
//     "@.str.annotation" (cache.empty() branch). It happens to hold
//     "static_local" given collection order (globals first), so we pin
//     both the name and the content here.
// LLVM-DAG: @.str.annotation = private unnamed_addr constant {{.*}} c"static_local\00", section "llvm.metadata"
//
// (b) Subsequent name strings get a numeric index (cache.size() branch).
//     The second-emitted string is the source-file path, shared by every
//     annotation entry below (string cache HIT path).
// LLVM-DAG: @.str.1.annotation = private unnamed_addr constant {{.*}} c"{{.*annotate-attribute\.c}}\00", section "llvm.metadata"
//
// (c) Every other annotation name we wrote survives as its own
//     llvm.metadata string. Each line targets a distinct emitted
//     constant (static_local is covered above by (a)).
// LLVM-DAG: constant {{.*}} c"introduced_in=29\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"test_annotation\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"api_level=29\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"ann1\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"ann2\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"with_args\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"uniq\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"inherited_decl_ann\00", section "llvm.metadata"
// LLVM-DAG: constant {{.*}} c"inherited_def_ann\00", section "llvm.metadata"
//
// (d) String args take the isArg=true branch of
//     getOrCreateAnnotationStringGlobal: ".annotation.arg" suffix,
//     alignment 1, and NO llvm.metadata section. Both the no-index and
//     numbered name forms are present.
// LLVM-DAG: @.str.annotation.arg = private unnamed_addr constant [{{[0-9]+}} x i8] c"str_arg\00", align 1
// LLVM-DAG: @.str.1.annotation.arg = private unnamed_addr constant [{{[0-9]+}} x i8] c"shared\00", align 1
//
// (e) Args struct globals follow @.args{,.<n>}.annotation: first one
//     uses the no-index branch (argsCache.empty()), subsequent ones
//     get a numeric suffix. Both branches are exercised here.
// LLVM-DAG: @.args.annotation = private unnamed_addr constant { ptr, i32 } { ptr @.str{{(\.[0-9]+)?}}.annotation.arg, i32 42 }, section "llvm.metadata"
// LLVM-DAG: @.args.1.annotation = private unnamed_addr constant { ptr, i32 } { ptr @.str{{(\.[0-9]+)?}}.annotation.arg, i32 7 }, section "llvm.metadata"
//
// (f) The @llvm.global.annotations array is appending and lives in
//     llvm.metadata. uniq_a and uniq_b reference the SAME args global
//     (@.args.1.annotation) — the args-cache HIT path.
// LLVM-DAG: @llvm.global.annotations = appending global [11 x { ptr, ptr, ptr, i32, ptr }] [{{.*}} { ptr @uniq_a, ptr {{[^,]+}}, ptr {{[^,]+}}, i32 {{[0-9]+}}, ptr @.args.1.annotation }, {{.*}} { ptr @uniq_b, ptr {{[^,]+}}, ptr {{[^,]+}}, i32 {{[0-9]+}}, ptr @.args.1.annotation }{{.*}}], section "llvm.metadata"

__attribute__((annotate("test_annotation")))
void annotated_func(void) {}

// CIR: cir.func {{.*}} @annotated_func() [#cir.annotation<"test_annotation">]
// LLVM: define{{.*}} void @annotated_func()
// OGCG: define{{.*}} void @annotated_func()

// Test: annotated function declaration used before definition exercises the
// deferred annotation path in emitGlobal.
__attribute__((annotate("api_level=29")))
void deferred_annotated(void);

void caller(void) {
  deferred_annotated();
}

__attribute__((annotate("api_level=29")))
void deferred_annotated(void) {}

// CIR: cir.func {{.*}} @deferred_annotated() [#cir.annotation<"api_level=29">]
// LLVM: define{{.*}} void @deferred_annotated()
// OGCG: define{{.*}} void @deferred_annotated()

// Test: multiple annotations on a single function.
__attribute__((annotate("ann1")))
__attribute__((annotate("ann2")))
void multi_annotated(void) {}

// CIR: cir.func {{.*}} @multi_annotated() [#cir.annotation<"ann1">, #cir.annotation<"ann2">]
// LLVM: define{{.*}} void @multi_annotated()
// OGCG: define{{.*}} void @multi_annotated()

// Test: annotation with arguments.
__attribute__((annotate("with_args", "str_arg", 42)))
void annotated_with_args(void) {}

// CIR: cir.func {{.*}} @annotated_with_args() [#cir.annotation<"with_args", ["str_arg", 42 : i32]>]
// LLVM: define{{.*}} void @annotated_with_args()
// OGCG: define{{.*}} void @annotated_with_args()

// Test: function-local static variable with annotation. Exercises the
// emitStaticVarDecl -> addGlobalAnnotations path (distinct from
// emitGlobalVarDefinition for plain globals). The CIR-DAG / LLVM-DAG
// CHECKs for the emitted global live in the globals block at the top.
void with_static_annotated(void) {
  static int counter __attribute__((annotate("static_local"))) = 0;
  counter++;
}

// Test: argument-uniquing path in emitAnnotationArgs. Two annotations with
// identical arg lists should share the same args ArrayAttr (the FoldingSet
// cache hit). MLIR uniques attributes on identity, so identical printed
// output here is the observable signal that uniquing took effect.
__attribute__((annotate("uniq", "shared", 7)))
void uniq_a(void) {}

__attribute__((annotate("uniq", "shared", 7)))
void uniq_b(void) {}

// CIR: cir.func {{.*}} @uniq_a() [#cir.annotation<"uniq", ["shared", 7 : i32]>]
// CIR: cir.func {{.*}} @uniq_b() [#cir.annotation<"uniq", ["shared", 7 : i32]>]
// OGCG: define{{.*}} void @uniq_a()
// OGCG: define{{.*}} void @uniq_b()

// Test: declaration annotated, then defined annotated, with a forward use
// in between. This exercises the emitGlobal "update deferred annotations
// with the latest declaration" path: the call site triggers
// getOrCreateCIRFunction (records the decl), then the definition's
// emitGlobal sees the existing GlobalValue and updates the deferred map
// with the *defining* decl so all inherited annotations stick.
__attribute__((annotate("inherited_decl_ann")))
void inherited(void);

void inherited_caller(void) { inherited(); }

__attribute__((annotate("inherited_def_ann")))
void inherited(void) {}

// Both annotations should appear (decl + def).
// CIR: cir.func {{.*}} @inherited() [#cir.annotation<"inherited_decl_ann">, #cir.annotation<"inherited_def_ann">]
// OGCG: define{{.*}} void @inherited()
