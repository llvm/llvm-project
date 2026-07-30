// Test Behavior of -mloadtime-comment-vars= for file-scope variables, in C and in
// the same source compiled as C++:
//   * supported forms (plain-char pointer/array with a string-literal
//     initializer) named in the list get !loadtime_comment metadata and are
//     kept alive in llvm.compiler.used, even when otherwise unreferenced;
//   * unsupported or unlisted variables are silently ignored and, when
//     unreferenced, are not emitted at all;
//   * on non-AIX targets the option has no effect;
//   * names are matched against the mangled IR name: in C that is the source
//     identifier; in C++ a file-scope static mangles (sccsid -> _ZL6sccsid).


// C, 32-bit and 64-bit AIX.
// RUN: %clang_cc1 -O2 -triple powerpc-ibm-aix -mloadtime-comment-vars=sccsid,version,build_number,same_copyright,active,not_defined_here,tdefchar,ustr,sstr -emit-llvm -disable-llvm-passes -o %t-c32.ll %s
// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix -mloadtime-comment-vars=sccsid,version,build_number,same_copyright,active,not_defined_here,tdefchar,ustr,sstr -emit-llvm -disable-llvm-passes -o %t-c64.ll %s
// RUN: FileCheck %s -DSCCSID=sccsid -DVERSION=version -DSAME=same_copyright -DACTIVE=active -DTYPEDEFCHAR=tdefchar < %t-c32.ll
// RUN: FileCheck %s -DSCCSID=sccsid -DVERSION=version -DSAME=same_copyright -DACTIVE=active -DTYPEDEFCHAR=tdefchar < %t-c64.ll
// RUN: FileCheck %s --check-prefix=NOEMIT -DCOPYRIGHT=copyright -DBUILDNUM=build_number -DBUILDDATA=build_data -DUSTR=ustr -DSSTR=sstr < %t-c32.ll
// RUN: FileCheck %s --check-prefix=NOEMIT -DCOPYRIGHT=copyright -DBUILDNUM=build_number -DBUILDDATA=build_data -DUSTR=ustr -DSSTR=sstr < %t-c64.ll

// The same source as C++: internal-linkage statics are matched by mangled
// name. (-w silences the C++ writable-strings compatibility warning for the
// legacy `static char *` idiom.)
// RUN: %clang_cc1 -x c++ -w -O2 -triple powerpc64-ibm-aix -mloadtime-comment-vars=_ZL6sccsid,_ZL7version,_ZL12build_number,_ZL14same_copyright,_ZL6active,not_defined_here,_ZL8tdefchar,_ZL4ustr,_ZL4sstr -emit-llvm -disable-llvm-passes -o %t-cxx.ll %s
// RUN: FileCheck %s -DSCCSID=_ZL6sccsid -DVERSION=_ZL7version -DSAME=_ZL14same_copyright -DACTIVE=_ZL6active -DTYPEDEFCHAR=_ZL8tdefchar < %t-cxx.ll
// RUN: FileCheck %s --check-prefix=NOEMIT -DCOPYRIGHT=_ZL9copyright -DBUILDNUM=_ZL12build_number -DBUILDDATA=_ZL10build_data -DUSTR=_ZL4ustr -DSSTR=_ZL4sstr < %t-cxx.ll

// Non-AIX target: the option has no effect.
// RUN: %clang_cc1 -O2 -triple x86_64-linux-gnu -mloadtime-comment-vars=sccsid,version -emit-llvm -disable-llvm-passes -o %t-linux.ll %s
// RUN: FileCheck %s --check-prefix=LINUX < %t-linux.ll
// RUN: FileCheck %s --check-prefix=NOEMIT -DCOPYRIGHT=copyright -DBUILDNUM=build_number -DBUILDDATA=build_data -DUSTR=ustr -DSSTR=sstr < %t-linux.ll
// RUN: FileCheck %s --check-prefix=NOPRESERVE < %t-linux.ll

// 1. A string pointer named in the list is preserved.
static char *sccsid = "@(#) sccsid Version 1.0";

// 2. A string array named in the list is preserved.
static char version[] = "@(#) Copyright Version 2.0";

// 3. Const string (not in the list; unreferenced, so not emitted)
static const char *copyright = "@(#) Copyright 2026";

// 4. Integer (in the list but unsupported type; not emitted)
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
//     are silently ignored and not emitted.
static unsigned char ustr[] = "@(#) unsigned char string";
static signed char sstr[] = "@(#) signed char string";

void foo() {}

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

// CHECK: @llvm.compiler.used = appending global [5 x ptr]
// CHECK-SAME: ptr @[[SCCSID]]
// CHECK-SAME: ptr @[[VERSION]]
// CHECK-SAME: ptr @[[SAME]]
// CHECK-SAME: ptr @[[ACTIVE]]
// CHECK-SAME: ptr @[[TYPEDEFCHAR]]
// CHECK-SAME: section "llvm.metadata"

// The unlisted const string, the unsupported-type variables (including the
// listed signed/unsigned char strings, whose element type is not plain char),
// and the extern declaration are not emitted in any configuration.
// NOEMIT-NOT: @[[COPYRIGHT]]
// NOEMIT-NOT: @[[BUILDNUM]]
// NOEMIT-NOT: @[[BUILDDATA]]
// NOEMIT-NOT: @[[USTR]]
// NOEMIT-NOT: @[[SSTR]]
// NOEMIT-NOT: @not_defined_here

// On non-AIX targets a referenced variable is still emitted normally, just
// not preserved.
// LINUX: @active = internal global ptr
// LINUX: define {{.*}}void @bar

// On non-AIX targets nothing is preserved: no metadata, and the unreferenced
// statics -- listed or not -- are not emitted.
// NOPRESERVE-NOT: loadtime_comment
// NOPRESERVE-NOT: @sccsid
// NOPRESERVE-NOT: @version
// NOPRESERVE-NOT: @same_copyright
