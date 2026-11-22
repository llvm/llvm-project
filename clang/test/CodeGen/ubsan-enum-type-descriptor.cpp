// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsanitize=enum -emit-llvm %s -o - | FileCheck %s

// The UBSan type descriptor wraps the size and signedness of the checked type
// so that the runtime can print the offending value. An enum does have the same
// size and value representation as its underlying type, so the descriptor has
// to unwrap up to the underlying type and not the enum: querying the enum type
// itself leaves a scoped enum as TK_Unknown (i16 -1) with no width, and the
// runtime then prints "<unknown>" instead of the value.

// A scoped enum has a fixed underlying type, so it only reaches the range check
// when that type has values outside the enumerator range, i.e. bool.
enum class EBool : bool { a = 1 };

// TypeKind is TK_Integer (0) and TypeInfo is (log2(8) << 1) => 6.
// Name length includes the quotes (''), so 8
// CHECK: @{{[0-9]+}} = private unnamed_addr constant { i16, i16, [8 x i8] } { i16 0, i16 6, [8 x i8] c"'EBool'\00" }

enum EPlain { p = 1 };

// TypeInfo is (log2(32) << 1) ==> 10.
// CHECK: @{{[0-9]+}} = private unnamed_addr constant { i16, i16, [9 x i8] } { i16 0, i16 10, [9 x i8] c"'EPlain'\00" }

// A negative enumerator gives the enum a signed underlying type, which sets the
// low bit of TypeInfo.
enum ENeg { n = -1 };

// TypeInfo is ((log2(32) << 1) | 1) ==> 11.
// CHECK: @{{[0-9]+}} = private unnamed_addr constant { i16, i16, [7 x i8] } { i16 0, i16 11, [7 x i8] c"'ENeg'\00" }

// Descriptors are only emitted where a check is, so force a checked load of
// each

bool load_scoped(EBool *p) { return (bool)*p; }
// CHECK: call void @__ubsan_handle_load_invalid_value_abort

int load_unscoped(EPlain *p) { return (int)*p; }
// CHECK: call void @__ubsan_handle_load_invalid_value_abort

int load_signed(ENeg *p) { return (int)*p; }
// CHECK: call void @__ubsan_handle_load_invalid_value_abort
