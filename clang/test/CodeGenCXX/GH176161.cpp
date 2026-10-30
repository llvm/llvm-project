// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -emit-llvm -o - %s | FileCheck %s

template <typename T> struct ConfigFlag { T value; };

struct Tuple {
  Tuple(ConfigFlag<bool> &&a, ConfigFlag<bool> &&b) : first(a), second(b) {}
  ConfigFlag<bool> first, second;
};

struct ConfigSystem {
  Tuple flags;
};

template <unsigned R> class M {
public:
  M(const int *arr) {
    auto flagProcessing = [] {
      struct Configurator {
        ConfigSystem config;
        Configurator()
            : config({ConfigFlag<bool>(true), ConfigFlag<bool>(false)}) {}
      };
      Configurator configurator;
    };
    flagProcessing();
    for (unsigned long r = 0; r < R; ++r)
      m[r] = arr[r];
  }
  int m[R];
};

int main() {
  int arr[2] = {1, 2};
  M<2> m(arr);
}

// CHECK-LABEL: define {{.*}}12ConfiguratorC2Ev(
// CHECK: store i8 1, ptr
// CHECK: store i8 0, ptr
// CHECK: call void @_ZN5TupleC{{[12]}}E
// CHECK: ret void

namespace GH213284 {
struct Ref { unsigned long long bits; };
template <typename> struct Result {
  Result() : thing(0) {}
  Ref thing;
};
Result<void> construct() { return Result<void>(); }
}

// CHECK-LABEL: define {{.*}}@_ZN8GH2132846ResultIvEC2Ev(
// CHECK: store i64 0, ptr
