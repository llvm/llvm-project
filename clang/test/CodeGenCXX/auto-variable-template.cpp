// RUN: %clang_cc1 -no-opaque-pointers -std=c++14 %s -triple=x86_64-linux -emit-llvm -o - | FileCheck %s

struct f {
  void operator()() const {}
};

template <typename T> auto vtemplate = f{};

int main() { vtemplate<int>(); }

// CHECK: @_Z9vtemplateIiE = linkonce_odr global %struct.f undef, comdat

// CHECK: define{{.*}} i32 @main()
// CHECK: call void @_ZNK1fclEv(%struct.f* {{[^,]*}} @_Z9vtemplateIiE)

template <typename>
struct pack {
  template <typename T>
  constexpr static auto some_boolean_cx_value = true;
};

auto usage() {
  return pack<char>::some_boolean_cx_value<int>;
}

// CHECK: define{{.*}} i1 @_Z5usagev()

auto otherusage() {
  return pack<char>{}.some_boolean_cx_value<int>;
}

// CHECK: define{{.*}} i1 @_Z10otherusagev()
