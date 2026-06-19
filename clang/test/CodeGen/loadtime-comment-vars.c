// RUN: %clang_cc1 -O2 -triple powerpc-ibm-aix -mloadtime-comment-vars=sccsid,version,build_number,same_copyright,active,not_defined_here -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -O2 -triple powerpc64-ibm-aix -mloadtime-comment-vars=sccsid,version,build_number,same_copyright,active,not_defined_here -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// RUN: %clang_cc1 -O2 -triple x86_64-linux-gnu -mloadtime-comment-vars=sccsid,version -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=NONAIX

// 1. String pointer 
static char *sccsid = "@(#) sccsid Version 1.0";

// 2. String array 
static char version[] = "@(#) Copyright Version 2.0";

// 3. Const string (Not in CLI list, should NOT be emitted)
static const char *copyright = "@(#) Copyright 2026";

// 4. Integer (In CLI list but invalid type, should NOT be emitted)
static int build_number = 12345;

// 5. Struct (not in CLI list and invalid type, NOT emitted)
struct build_info {
    int major;
    int minor;
} static build_data = {1, 0};

// 6. Deferred: pointer whose initializer references another static global.
// Both the pointer AND the string it points to must be emitted.
static const char dummy[] = "dummy copyright deferred";
static const char *same_copyright = dummy;

// 7. Variable already referenced (eager emission path)
static char *active = "@(#) active string";
void bar() { (void)active; }

// 8. Variable listed but only declared (extern)
extern char *not_defined_here;

void foo() {}

// CHECK-DAG: @active = internal global ptr @.str, align {{[0-9]+}}, !loadtime_comment ![[MD:[0-9]+]]
// CHECK: @.str = private unnamed_addr constant [19 x i8] c"@(#) active string\00", align {{[0-9]+}}
// CHECK-DAG: @sccsid = internal global ptr @.str.1, align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK: @.str.1 = private unnamed_addr constant [24 x i8] c"@(#) sccsid Version 1.0\00", align {{[0-9]+}}
// CHECK-DAG: @version = internal global [27 x i8] c"@(#) Copyright Version 2.0\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @same_copyright = internal global ptr @dummy, align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK: @dummy = internal constant [25 x i8] c"dummy copyright deferred\00"
// CHECK: @llvm.compiler.used = appending global [4 x ptr]
// CHECK-SAME: ptr @sccsid
// CHECK-SAME: ptr @version
// CHECK-SAME: ptr @same_copyright
// CHECK-SAME: ptr @active
// CHECK-SAME: section "llvm.metadata"

// Ensure unrequested/invalid variables are not emitted
// CHECK-NOT: @copyright
// CHECK-NOT: @build_number
// CHECK-NOT: @build_data
// CHECK-NOT: @not_defined_here

// NONAIX-NOT: loadtime_comment
// NONAIX-NOT: @sccsid
// NONAIX-NOT: @version

