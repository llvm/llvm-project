// Here we test the order of the Checkers when StdCLibraryFunctions is
// enabled.

// RUN: %clang --analyze %s --target=x86_64-pc-linux-gnu \
// RUN:   -Xclang -analyzer-checker=core \
// RUN:   -Xclang -analyzer-checker=alpha.unix.StdCLibraryFunctions \
// RUN:   -Xclang -analyzer-config \
// RUN:      -Xclang alpha.unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -Xclang -analyzer-checker=alpha.unix.Stream \
// RUN:   -Xclang -analyzer-list-enabled-checkers \
// RUN:   -Xclang -analyzer-display-progress \
// RUN:   2>&1 | FileCheck %s --implicit-check-not=ANALYZE \
// RUN:                       --implicit-check-not=\.

// CHECK:      OVERVIEW: Clang Static Analyzer Enabled Checkers List
// CHECK-EMPTY:
// CHECK-NEXT: core.CallAndMessageModeling
// CHECK-NEXT: core.CallAndMessage
// CHECK-NEXT: core.NonNullParamChecker
// CHECK-NEXT: alpha.unix.Stream
// CHECK-NEXT: alpha.unix.StdCLibraryFunctions
// CHECK-NEXT: apiModeling.Errno
// CHECK-NEXT: apiModeling.TrustNonnull
// CHECK-NEXT: apiModeling.TrustReturnsNonnull
// CHECK-NEXT: apiModeling.llvm.CastValue
// CHECK-NEXT: apiModeling.llvm.ReturnValue
// CHECK-NEXT: core.BitwiseShift
// CHECK-NEXT: core.DivideZero
// CHECK-NEXT: core.DynamicTypePropagation
// CHECK-NEXT: core.NonnilStringConstants
// CHECK-NEXT: core.NullDereference
// CHECK-NEXT: core.StackAddrEscapeBase
// CHECK-NEXT: core.StackAddressEscape
// CHECK-NEXT: core.UndefinedBinaryOperatorResult
// CHECK-NEXT: core.VLASize
// CHECK-NEXT: core.builtin.BuiltinFunctions
// CHECK-NEXT: core.builtin.NoReturnFunctions
// CHECK-NEXT: core.uninitialized.ArraySubscript
// CHECK-NEXT: core.uninitialized.Assign
// CHECK-NEXT: core.uninitialized.Branch
// CHECK-NEXT: core.uninitialized.CapturedBlockVariable
// CHECK-NEXT: core.uninitialized.UndefReturn
// CHECK-NEXT: deadcode.DeadStores
// CHECK-NEXT: nullability.NullabilityBase
// CHECK-NEXT: nullability.NullPassedToNonnull
// CHECK-NEXT: nullability.NullReturnedFromNonnull
// CHECK-NEXT: security.insecureAPI.SecuritySyntaxChecker
// CHECK-NEXT: security.insecureAPI.UncheckedReturn
// CHECK-NEXT: security.insecureAPI.getpw
// CHECK-NEXT: security.insecureAPI.gets
// CHECK-NEXT: security.insecureAPI.mkstemp
// CHECK-NEXT: security.insecureAPI.mktemp
// CHECK-NEXT: security.insecureAPI.vfork
// CHECK-NEXT: unix.API
// CHECK-NEXT: unix.cstring.CStringModeling
// CHECK-NEXT: unix.DynamicMemoryModeling
// CHECK-NEXT: unix.Malloc
// CHECK-NEXT: unix.MallocSizeof
// CHECK-NEXT: unix.MismatchedDeallocator
// CHECK-NEXT: unix.Vfork
// CHECK-NEXT: unix.cstring.BadSizeArg
// CHECK-NEXT: unix.cstring.NullArg

int main() {
  int i;
  (void)(10 / i);
}
