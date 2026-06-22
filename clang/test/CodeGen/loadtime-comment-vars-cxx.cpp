// RUN: %clang_cc1 -std=c++17 -O2 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=x,N::q,A::x,N::ptr,B::ver,C::info \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s


// 1. Unqualified name "x" — matches both ::x (file scope) and N::x (namespace)
char x[] = "@(#) global x";

namespace N {
char x[] = "@(#) ns x";

// 2. Qualified name "N::q" — selects only this declaration
char q[] = "@(#) ns q";


// 3. Deferred pointer-chain inside a namespace.
//    N::ptr points to N::base (another static). MustBeEmitted forces N::ptr
//    through EmitGlobalVarDefinition; the initializer reference to N::base
//    causes N::base to be emitted as a side-effect.
static const char base[] = "base deferred ns";
static const char *ptr = base;
} // namespace N


// 4. Qualified name "A::x" — class static member (const char *)
struct A {
  static const char *x;
};
const char *A::x = "@(#) class x";


// 5. Deferred pointer-chain for a class static member.
//    B::ver points to a separate static array base_b.
struct B {
  static const char *ver;
};
static const char base_b[] = "base for B::ver";
const char *B::ver = base_b;

// 6. Qualified name in list but only declared, never defined — must be skipped.
struct C { static const char *info; };
// C::info has no definition in this TU.


// 7. Invalid type — int with a matching name should NOT be tagged.
int not_string = 7;

void f() {}

// --- Checks ----------------------------------------------------------------

// Unqualified "x" matches both ::x and N::x.
// CHECK-DAG: @x = global [14 x i8] c"@(#) global x\00", align {{[0-9]+}}, !loadtime_comment ![[MD:[0-9]+]]
// CHECK-DAG: @_ZN1N1xE = global [10 x i8] c"@(#) ns x\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// Qualified "N::q" selects the specific namespace member.
// CHECK-DAG: @_ZN1N1qE = global [10 x i8] c"@(#) ns q\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// Qualified "A::x" selects the class static member (pointer to literal).
// CHECK-DAG: @[[AX:_ZN1A1xE]] = {{.*}}global ptr @[[AXSTR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[AXSTR]] = private unnamed_addr constant [13 x i8] c"@(#) class x\00", align {{[0-9]+}}

// Deferred: N::ptr points to N::base — both must be emitted.
// CHECK-DAG: @_ZN1NL3ptrE = internal global ptr @_ZN1NL4baseE, align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @_ZN1NL4baseE = internal constant [17 x i8] c"base deferred ns\00", align {{[0-9]+}}

// Deferred: B::ver points to base_b — both must be emitted.
// CHECK-DAG: @_ZN1B3verE = global ptr @_ZL6base_b, align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @_ZL6base_b = internal constant [16 x i8] c"base for B::ver\00", align {{[0-9]+}}

// Invalid type must not be tagged.
// CHECK-NOT: @not_string{{.*}}!loadtime_comment

// C::info is declared but not defined — must not appear at all.
// CHECK-NOT: @_ZN1C4infoE

// All six selected globals are preserved in llvm.compiler.used.
// CHECK: @llvm.compiler.used = appending global [6 x ptr]
// CHECK-SAME: @x
// CHECK-SAME: @_ZN1N1xE
// CHECK-SAME: @_ZN1N1qE
// CHECK-SAME: @_ZN1NL3ptrE
// CHECK-SAME: @[[AX]]
// CHECK-SAME: @_ZN1B3verE
// CHECK-SAME: section "llvm.metadata"
