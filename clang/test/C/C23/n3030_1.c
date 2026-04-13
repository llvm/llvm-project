// RUN: %clang_cc1 -std=c23 -Wno-underlying-atomic-qualifier-ignored -ast-dump %s | FileCheck %s

// The underlying type is the unqualified, non-atomic version of the type
// specified.
enum const_enum : const short { ConstE };
// CHECK: EnumDecl {{.*}} const_enum 'short'

// These were previously being diagnosed as invalid underlying types. They
// are valid; the _Atomic is stripped from the underlying type.
enum atomic_enum1 : _Atomic(int) { AtomicE1 };
// CHECK: EnumDecl {{.*}} atomic_enum1 'int'
enum atomic_enum2 : _Atomic long long { AtomicE2 };
// CHECK: EnumDecl {{.*}} atomic_enum2 'long long'
