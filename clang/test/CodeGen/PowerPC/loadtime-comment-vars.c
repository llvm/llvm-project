// Test the behavior of -mloadtime-comment-vars= for file-scope variables, in
// C and in the same source compiled as C++ (CHECK and NOEMIT):
//   * supported forms (plain-char pointer/array with a string-literal
//     initializer) named in the list get !loadtime_comment metadata and are
//     kept alive in llvm.compiler.used, even when otherwise unreferenced;
//   * unlisted variables, and listed variables of an unsupported type, are
//     not preserved and, when unreferenced, are not emitted at all (the
//     unsupported-type diagnostic itself is covered by the Sema tests; it is
//     silenced here with -Wno-loadtime-comment-var);
//   * names are matched against the mangled IR name: in C that is the source
//     identifier; in C++ a file-scope static mangles (sccsid -> _ZL6sccsid).
//
//   * STORAGE — storage-duration filtering: a thread-local variable and a
//     function-local static receive no metadata (their diagnostics are
//     covered by the Sema tests and silenced here);
//   * SPACE/DUP — list parsing: a name with a leading space matches nothing,
//     and a duplicate name preserves the variable exactly once.
//
// Non-AIX behavior is covered elsewhere: the driver warns and drops the
// option (clang/test/Driver/mloadtime-comment-vars.c) and cc1 rejects it
// with an error (clang/test/Sema/loadtime-comment-vars.c).


// C, 32-bit and 64-bit AIX.
// RUN: %clang_cc1 -O2 -triple powerpc-ibm-aix -Wno-loadtime-comment-var -mloadtime-comment-vars=sccsid,version,build_number,same_copyright,active,not_defined_here,tdefchar,ustr,sstr,braced -emit-llvm -disable-llvm-passes -o %t-c32.ll %s
// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var -mloadtime-comment-vars=sccsid,version,build_number,same_copyright,active,not_defined_here,tdefchar,ustr,sstr,braced -emit-llvm -disable-llvm-passes -o %t-c64.ll %s
// RUN: FileCheck %s -DSCCSID=sccsid -DVERSION=version -DSAME=same_copyright -DACTIVE=active -DTYPEDEFCHAR=tdefchar -DBRACED=braced < %t-c32.ll
// RUN: FileCheck %s -DSCCSID=sccsid -DVERSION=version -DSAME=same_copyright -DACTIVE=active -DTYPEDEFCHAR=tdefchar -DBRACED=braced < %t-c64.ll
// RUN: FileCheck %s --check-prefix=NOEMIT -DCOPYRIGHT=copyright -DBUILDNUM=build_number -DBUILDDATA=build_data -DUSTR=ustr -DSSTR=sstr < %t-c32.ll
// RUN: FileCheck %s --check-prefix=NOEMIT -DCOPYRIGHT=copyright -DBUILDNUM=build_number -DBUILDDATA=build_data -DUSTR=ustr -DSSTR=sstr < %t-c64.ll

// The same source as C++: internal-linkage statics are matched by mangled
// name. (-w silences the C++ writable-strings compatibility warning for the
// legacy `static char *` idiom.)
// RUN: %clang_cc1 -x c++ -w -O2 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var -mloadtime-comment-vars=_ZL6sccsid,_ZL7version,_ZL12build_number,_ZL14same_copyright,_ZL6active,not_defined_here,_ZL8tdefchar,_ZL4ustr,_ZL4sstr,_ZL6braced -emit-llvm -disable-llvm-passes -o %t-cxx.ll %s
// RUN: FileCheck %s -DSCCSID=_ZL6sccsid -DVERSION=_ZL7version -DSAME=_ZL14same_copyright -DACTIVE=_ZL6active -DTYPEDEFCHAR=_ZL8tdefchar -DBRACED=_ZL6braced < %t-cxx.ll
// RUN: FileCheck %s --check-prefix=NOEMIT -DCOPYRIGHT=_ZL9copyright -DBUILDNUM=_ZL12build_number -DBUILDDATA=_ZL10build_data -DUSTR=_ZL4ustr -DSSTR=_ZL4sstr < %t-cxx.ll

// RUN: %clang_cc1 -triple powerpc64-ibm-aix -Wno-loadtime-comment-var -mloadtime-comment-vars=keep,tl,stl,fn -emit-llvm -o - %s | FileCheck %s --check-prefix=STORAGE

// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix "-mloadtime-comment-vars=one, two" -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=SPACE
// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix -mloadtime-comment-vars=one,one -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=DUP

// 1. A string pointer named in the list is preserved.
static char *sccsid = "@(#) sccsid Version 1.0";

// 2. A string array named in the list is preserved.
static char version[] = "@(#) Copyright Version 2.0";

// 3. Const string (not in the list; unreferenced, so not emitted)
static const char *copyright = "@(#) Copyright 2026";

// 4. Integer (in the list but unsupported type: diagnosed, not emitted)
static int build_number = 12345;

// 5. Struct (not in the list and unsupported type; not emitted)
struct build_info {
    int major;
    int minor;
} static build_data = {1, 0};

// 6. Pointer initialized with a string literal; forced into emission even
// though it is never referenced.
static const char *same_copyright = "@(#) same copyright";

// 7. Variable already referenced (eager emission path)
static char *active = "@(#) active string";
void bar() { (void)active; }

// 8. Variable listed but only declared (extern)
extern char *not_defined_here;

// 9. A typedef of plain char is looked through and matches like plain char.
typedef char CHAR;
static CHAR *tdefchar = "@(#) typedef char";

// 10. These are listed, but their element type is not plain char, so they
//     are diagnosed (see the Sema tests), not preserved, and not emitted.
static unsigned char ustr[] = "@(#) unsigned char string";
static signed char sstr[] = "@(#) signed char string";

// 11. A pointer whose string-literal initializer is enclosed in braces is
//     preserved like the unbraced form.
static const char *braced = {"@(#) braced"};

void foo() {}

// Sources for the STORAGE scenario. keep has static storage duration and is
// preserved; the thread-local variables and the function-local static (whose
// IR name is g.fn) are name-matched but receive no metadata.
const char *keep = "@(#) keep";
__thread const char *tl = "@(#) tl";
static __thread const char *stl = "@(#) stl";
void g(void) { static const char *fn = "@(#) fn"; (void)fn; }

// Sources for the SPACE and DUP scenarios.
char one[] = "@(#) one";
char two[] = "@(#) two";

// Listed, supported variables carry the metadata and stay alive.
// CHECK-DAG: @[[ACTIVE]] = internal global ptr @[[ACTIVE_STR:.str(\.[0-9]+)?]], align {{[0-9]+}}, !loadtime_comment ![[MD:[0-9]+]]
// CHECK-DAG: @[[ACTIVE_STR]] = private unnamed_addr constant [19 x i8] c"@(#) active string\00", align {{[0-9]+}}
// CHECK-DAG: @[[SCCSID]] = internal global ptr @[[SCCSID_STR:.str(\.[0-9]+)?]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[SCCSID_STR]] = private unnamed_addr constant [24 x i8] c"@(#) sccsid Version 1.0\00", align {{[0-9]+}}
// CHECK-DAG: @[[VERSION]] = internal global [27 x i8] c"@(#) Copyright Version 2.0\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[SAME]] = internal global ptr @[[SAME_STR:.str(\.[0-9]+)?]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[SAME_STR]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"@(#) same copyright\00", align {{[0-9]+}}
// CHECK-DAG: @[[TYPEDEFCHAR]] = internal global ptr @[[TYPEDEFCHAR_STR:.str(\.[0-9]+)?]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[TYPEDEFCHAR_STR]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"@(#) typedef char\00", align {{[0-9]+}}
// CHECK-DAG: @[[BRACED]] = internal global ptr @[[BRACED_STR:.str(\.[0-9]+)?]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[BRACED_STR]] = private unnamed_addr constant [{{[0-9]+}} x i8] c"@(#) braced\00", align {{[0-9]+}}

// CHECK: @llvm.compiler.used = appending global [6 x ptr]
// CHECK-SAME: ptr @[[SCCSID]]
// CHECK-SAME: ptr @[[VERSION]]
// CHECK-SAME: ptr @[[SAME]]
// CHECK-SAME: ptr @[[ACTIVE]]
// CHECK-SAME: ptr @[[TYPEDEFCHAR]]
// CHECK-SAME: ptr @[[BRACED]]
// CHECK-SAME: section "llvm.metadata"

// The unlisted const string, the unsupported-type variables (including the
// listed signed/unsigned char strings, whose element type is not plain char
// and which are therefore diagnosed rather than preserved), and the extern
// declaration are not emitted in any configuration.
// NOEMIT-NOT: @[[COPYRIGHT]]
// NOEMIT-NOT: @[[BUILDNUM]]
// NOEMIT-NOT: @[[BUILDDATA]]
// NOEMIT-NOT: @[[USTR]]
// NOEMIT-NOT: @[[SSTR]]
// NOEMIT-NOT: @not_defined_here

// Storage-duration filtering: only the file-scope static-duration variable
// has metadata and appears in llvm.compiler.used.
// STORAGE: @keep = {{.*}}!loadtime_comment
// STORAGE-NOT: @tl = {{.*}}!loadtime_comment
// STORAGE-NOT: @stl = {{.*}}!loadtime_comment
// STORAGE-NOT: @g.fn = {{.*}}!loadtime_comment
// STORAGE: @llvm.compiler.used = appending global [1 x ptr] [ptr @keep], section "llvm.metadata"

// List parsing, "one, two": the leading space means ' two' matches nothing;
// only one is preserved (the {{$}} anchor proves two has no metadata).
// SPACE-DAG: @one = global [9 x i8] c"@(#) one\00", align {{[0-9]+}}, !loadtime_comment !{{[0-9]+}}
// SPACE-DAG: @two = global [9 x i8] c"@(#) two\00", align {{[0-9]+}}{{$}}
// SPACE-DAG: @llvm.compiler.used = appending global [1 x ptr] [ptr @one], section "llvm.metadata"

// List parsing, "one,one": a duplicate name preserves the variable exactly
// once.
// DUP-DAG: @one = global [9 x i8] c"@(#) one\00", align {{[0-9]+}}, !loadtime_comment !{{[0-9]+}}
// DUP-DAG: @two = global [9 x i8] c"@(#) two\00", align {{[0-9]+}}{{$}}
// DUP-DAG: @llvm.compiler.used = appending global [1 x ptr] [ptr @one], section "llvm.metadata"
