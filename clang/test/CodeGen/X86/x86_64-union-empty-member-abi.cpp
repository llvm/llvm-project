// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s

struct Empty {};
struct alignas(16) EmptyAligned {};

extern "C" {

// The empty member's alignment outranks the int's, so it wins the union's
// storage-type comparison and the coercion widens to i64 unless members that
// supply no bytes are skipped.
union OverAligned { EmptyAligned e; int i; };
void take_over_aligned(union OverAligned u);
void call_over_aligned(union OverAligned u) { take_over_aligned(u); }
// CHECK-DAG: declare void @take_over_aligned(i32)

// At equal alignment the comparison falls to size, which an array of empty
// records wins without supplying any bytes.
union ArrOfEmpty { Empty a[2]; char c; };
void take_arr_of_empty(union ArrOfEmpty u);
void call_arr_of_empty(union ArrOfEmpty u) { take_arr_of_empty(u); }
// CHECK-DAG: declare void @take_arr_of_empty(i8)

// The same array spanning a whole eightbyte.
union Arr8OfEmpty { Empty a[8]; char c; };
void take_arr8_of_empty(union Arr8OfEmpty u);
void call_arr8_of_empty(union Arr8OfEmpty u) { take_arr8_of_empty(u); }
// CHECK-DAG: declare void @take_arr8_of_empty(i8)

// Where the other member does fill the eightbyte there is nothing to narrow.
union EmptyAndBytes { Empty e; char c[8]; };
void take_empty_and_bytes(union EmptyAndBytes u);
void call_empty_and_bytes(union EmptyAndBytes u) { take_empty_and_bytes(u); }
// CHECK-DAG: declare void @take_empty_and_bytes(i64)

// Skipping the empty member leaves the remaining member's class intact.
union EmptyAndDouble { Empty e; double d; };
void take_empty_and_double(union EmptyAndDouble u);
void call_empty_and_double(union EmptyAndDouble u) { take_empty_and_double(u); }
// CHECK-DAG: declare void @take_empty_and_double(double)
}
