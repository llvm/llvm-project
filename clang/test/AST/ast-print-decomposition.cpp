// RUN: %clang_cc1 -std=c++26 -ast-print %s | FileCheck %s

// The `[a, b]` binding list must survive -ast-print, not just the type.

namespace std {
using size_t = decltype(sizeof(0));
template <typename> struct tuple_size;
template <size_t, typename> struct tuple_element;
} // namespace std

namespace Aggregate {
struct Pair { int a, b; };
Pair get();
Pair &getref();

// CHECK-LABEL: void local() {
void local() {
  // CHECK-NEXT: auto [x, y] = get();
  auto [x, y] = get();
  // CHECK-NEXT: auto & [rx, ry] = getref();
  auto &[rx, ry] = getref();
  // CHECK-NEXT: const auto [cx, cy] = get();
  const auto [cx, cy] = get();
  // CHECK-NEXT: static auto [sx, sy] = get();
  static auto [sx, sy] = get();
}
} // namespace Aggregate

namespace Array {
// CHECK-LABEL: void local() {
void local() {
  // CHECK-NEXT: int arr[3] = {1, 2, 3};
  int arr[3] = {1, 2, 3};
  // CHECK-NEXT: auto [a, b, c] = {arr[*]};
  auto [a, b, c] = arr;
}
} // namespace Array

namespace TupleLike {
struct Two {};
Two getTwo();
} // namespace TupleLike

template <> struct std::tuple_size<TupleLike::Two> { enum { value = 2 }; };
template <> struct std::tuple_element<0, TupleLike::Two> { using type = int; };
template <> struct std::tuple_element<1, TupleLike::Two> { using type = int; };

namespace TupleLike {
// get() must be found by ADL, so it needs to live here, not at global scope.
template <std::size_t N> int get(Two);

// CHECK-LABEL: void local() {
void local() {
  // CHECK-NEXT: auto [p, q] = getTwo();
  auto [p, q] = getTwo();
}
} // namespace TupleLike

namespace NamespaceScope {
using Aggregate::Pair;
using Aggregate::get;

// CHECK-NOT: {{^[[:space:]]*;[[:space:]]*$}}
// CHECK: auto [gx, gy] = get();
auto [gx, gy] = get();
// CHECK-NOT: {{^[[:space:]]*;[[:space:]]*$}}
} // namespace NamespaceScope

namespace Packs {
// CHECK-LABEL: void local() {
template <unsigned N> void local() {
  // CHECK-NEXT: int arr[4] = {1, 2, 3, 4};
  int arr[4] = {1, 2, 3, 4};
  // CHECK-NEXT: auto [first, ...rest, last] = {arr[*]};
  auto [first, ...rest, last] = arr;
}
void (*p)() = local<0>;
} // namespace Packs

namespace Attributes {
using Aggregate::Pair;
using Aggregate::get;

// CHECK-LABEL: void local() {
void local() {
  // CHECK-NEXT: auto [x {{\[\[}}maybe_unused]], y] = get();
  auto [x [[maybe_unused]], y] = get();
}
} // namespace Attributes

namespace OwnedTag {
// At declaration-context scope the printer groups a tag declaration with the
// declarators that follow it, so an owned tag type keeps `struct Owned { ... }
// obj;` on one line. A decomposition declaration is a declaration of its own
// and must never be pulled into that group, even though its deduced type is
// precisely the tag type owned by `obj`'s declaration.
// CHECK-LABEL: struct Owned {
// CHECK: } obj;
// CHECK-NEXT: auto [ox, oy] = obj;
struct Owned { int a, b; } obj;
auto [ox, oy] = obj;

// A second declarator of an owned tag type still merges.
// CHECK-NEXT: struct Merged {
// CHECK: } m1, m2;
struct Merged { int v; } m1, m2;
} // namespace OwnedTag
