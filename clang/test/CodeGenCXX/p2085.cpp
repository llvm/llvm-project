// RUN: %clang_cc1 --std=c++20 %s -emit-llvm -o - -triple x86_64-linux | FileCheck %s

namespace std {
struct strong_ordering {
  int n;
  constexpr operator int() const { return n; }
  static const strong_ordering equal, greater, less;
};
constexpr inline strong_ordering strong_ordering::equal = {0};
constexpr inline strong_ordering strong_ordering::greater = {1};
constexpr inline strong_ordering strong_ordering::less = {-1};
} // namespace std

struct Space {
  int i, j;

  std::strong_ordering operator<=>(Space const &other) const;
  bool operator==(Space const &other) const;
};

// Make sure these cause emission
std::strong_ordering Space::operator<=>(Space const &other) const = default;
// CHECK-LABEL: define{{.*}} @_ZNK5SpacessERKS_
bool Space::operator==(Space const &) const = default;
// CHECK-LABEL: define{{.*}} @_ZNK5SpaceeqERKS_

struct Water {
  int i, j;

  std::strong_ordering operator<=>(Water const &other) const;
  bool operator==(Water const &other) const;
};

// Make sure these do not cause emission
inline std::strong_ordering Water::operator<=>(Water const &other) const = default;
// CHECK-NOT: define{{.*}} @_ZNK5WaterssERKS_
inline bool Water::operator==(Water const &) const = default;
// CHECK-NOT: define{{.*}} @_ZNK5WatereqERKS_
