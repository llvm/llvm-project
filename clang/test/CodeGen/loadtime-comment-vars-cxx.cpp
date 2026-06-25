// Names are matched against mangled IR symbol names.
// C++ variables use Itanium ABI mangling; C/file-scope statics keep their
// source name.
//
// Mangled names used here:
//   x            -> x             (file-scope, no mangling)
//   N::x         -> _ZN1N1xE
//   N::q         -> _ZN1N1qE
//   N::ptr       -> _ZN1NL3ptrE   (static, internal linkage)
//   A::x         -> _ZN1A1xE
//   B::ver       -> _ZN1B3verE
//   C::info      -> _ZN1C4infoE   (declared only, no definition — skipped)

// RUN: %clang_cc1 -std=c++17 -O2 -triple powerpc64-ibm-aix \
// RUN:   -mloadtime-comment-vars=x,_ZN1N1xE,_ZN1N1qE,_ZN1NL3ptrE,_ZN1A1xE,_ZN1B3verE,_ZN1C4infoE \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// 1. File-scope array "x" — no mangling in C++, IR name == source name.
char x[] = "@(#) global x";

namespace N {
char x[] = "@(#) ns x";

// 2. Namespace member "N::x" — mangled as _ZN1N1xE.
char q[] = "@(#) ns q";

// 3. Deferred pointer-chain inside a namespace.
//    _ZN1NL3ptrE (N::ptr) points to _ZN1NL4baseE (N::base, another static).
//    MustBeEmitted forces N::ptr through EmitGlobalVarDefinition; the
//    initializer reference to N::base causes N::base to be emitted too.
static const char base[] = "base deferred ns";
static const char *ptr = base;
} // namespace N

// 4. Class static member "A::x" — mangled as _ZN1A1xE.
struct A {
  static const char *x;
};
const char *A::x = "@(#) class x";

// 5. Deferred pointer-chain for a class static member.
//    _ZN1B3verE (B::ver) points to _ZL6base_b.
struct B {
  static const char *ver;
};
static const char base_b[] = "base for B::ver";
const char *B::ver = base_b;

// 6. _ZN1C4infoE is in the list but C::info has no definition in this TU —
//    must be silently skipped.
struct C { static const char *info; };

// 7. Invalid type — int must not be tagged regardless of its IR name.
int not_string = 7;

void f() {}

// --- Checks ----------------------------------------------------------------

// File-scope x and namespace N::x both matched.
// CHECK-DAG: @x = global [14 x i8] c"@(#) global x\00", align {{[0-9]+}}, !loadtime_comment ![[MD:[0-9]+]]
// CHECK-DAG: @_ZN1N1xE = global [10 x i8] c"@(#) ns x\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// N::q matched by mangled name _ZN1N1qE.
// CHECK-DAG: @_ZN1N1qE = global [10 x i8] c"@(#) ns q\00", align {{[0-9]+}}, !loadtime_comment ![[MD]]

// A::x matched by mangled name _ZN1A1xE.
// CHECK-DAG: @[[AX:_ZN1A1xE]] = {{.*}}global ptr @[[AXSTR:.*]], align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @[[AXSTR]] = private unnamed_addr constant [13 x i8] c"@(#) class x\00", align {{[0-9]+}}

// Deferred: N::ptr (_ZN1NL3ptrE) points to N::base (_ZN1NL4baseE).
// CHECK-DAG: @_ZN1NL3ptrE = internal global ptr @_ZN1NL4baseE, align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @_ZN1NL4baseE = internal constant [17 x i8] c"base deferred ns\00", align {{[0-9]+}}

// Deferred: B::ver (_ZN1B3verE) points to base_b (_ZL6base_b).
// CHECK-DAG: @_ZN1B3verE = global ptr @_ZL6base_b, align {{[0-9]+}}, !loadtime_comment ![[MD]]
// CHECK-DAG: @_ZL6base_b = internal constant [16 x i8] c"base for B::ver\00", align {{[0-9]+}}

// Invalid type must not be tagged.
// CHECK-NOT: @not_string{{.*}}!loadtime_comment

// C::info has no definition — must not appear.
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
