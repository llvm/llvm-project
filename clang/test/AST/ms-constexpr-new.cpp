// RUN: %clang_cc1 -fms-compatibility -fms-compatibility-version=19.33 -std=c++20 -ast-dump %s | FileCheck %s

// CHECK: used operator new
// CHECK: MSConstexprAttr 0x{{[0-9a-f]+}} <col:17, col:23>
[[nodiscard]] [[msvc::constexpr]] inline void* __cdecl operator new(decltype(sizeof(void*)), void* p) noexcept { return p; }

// CHECK: used constexpr construct_at
// CHECK: AttributedStmt 0x{{[0-9a-f]+}} <col:46, col:88>
// CHECK-NEXT: MSConstexprAttr 0x{{[0-9a-f]+}} <col:48, col:54>
// CHECK-NEXT: ReturnStmt 0x{{[0-9a-f]+}} <col:66, col:88>
constexpr int* construct_at(int* p, int v) { [[msvc::constexpr]] return ::new (p) int(v); }
constexpr bool check_construct_at() { int x; return *construct_at(&x, 42) == 42; }
static_assert(check_construct_at());
