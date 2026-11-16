// RUN: %clang_cc1 -triple aarch64-windows-msvc -std=c++20 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-windows-msvc -std=c++20 -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"

// Each record holds 16 bytes of doubles. An HFA is passed as [2 x double] and
// a record that MSVC does not treat as an HFA is passed as [2 x i64].

struct Empty {};
struct ZeroLength {
  int arr[0];
};
struct DoubleBase {
  double a;
};

extern "C" {

struct Pod {
  double a, b;
};
void arg_pod(Pod s) {}
// CHECK: define{{.*}} void @arg_pod([2 x double] %{{.*}})

struct TrivialCopyAssign {
  double a, b;
  TrivialCopyAssign &operator=(const TrivialCopyAssign &) = default;
};
void arg_trivial_copy_assign(TrivialCopyAssign s) {}
// CHECK: define{{.*}} void @arg_trivial_copy_assign([2 x double] %{{.*}})

struct NonTrivialCopyAssign {
  double a, b;
  NonTrivialCopyAssign &operator=(const NonTrivialCopyAssign &);
};
void arg_non_trivial_copy_assign(NonTrivialCopyAssign s) {}
// CHECK: define{{.*}} void @arg_non_trivial_copy_assign([2 x i64] %{{.*}})

struct NonDefaultCtor {
  double a, b;
  NonDefaultCtor(int);
};
void arg_non_default_ctor(NonDefaultCtor s) {}
// CHECK: define{{.*}} void @arg_non_default_ctor([2 x double] %{{.*}})

struct UserDefaultCtor {
  double a, b;
  UserDefaultCtor();
};
void arg_user_default_ctor(UserDefaultCtor s) {}
// CHECK: define{{.*}} void @arg_user_default_ctor([2 x i64] %{{.*}})

struct MemberInit {
  double a = 1.0;
  double b;
};
void arg_member_init(MemberInit s) {}
// CHECK: define{{.*}} void @arg_member_init([2 x i64] %{{.*}})

struct TrivialDtor {
  double a, b;
  ~TrivialDtor() = default;
};
void arg_trivial_dtor(TrivialDtor s) {}
// CHECK: define{{.*}} void @arg_trivial_dtor([2 x double] %{{.*}})

struct NonTrivialDtor {
  double a, b;
  ~NonTrivialDtor();
};
void arg_non_trivial_dtor(NonTrivialDtor s) {}
// CHECK: define{{.*}} void @arg_non_trivial_dtor([2 x i64] %{{.*}})

// A polymorphic record has a non-trivial copy constructor, so it is passed
// indirectly and is never classified as an HFA.
struct Polymorphic {
  double a, b;
  virtual void f();
};
void arg_polymorphic(Polymorphic s) {}
// CHECK: define{{.*}} void @arg_polymorphic(ptr {{.*}}dereferenceable(24) %{{.*}})

struct PrivateMember {
  double a;

private:
  double b;
};
void arg_private_member(PrivateMember s) {}
// CHECK: define{{.*}} void @arg_private_member([2 x double] %{{.*}})

// A non-empty base contributes members, but an empty base disqualifies the
// derived record.
struct NonEmptyBase : DoubleBase {
  double b;
};
void arg_non_empty_base(NonEmptyBase s) {}
// CHECK: define{{.*}} void @arg_non_empty_base([2 x double] %{{.*}})

struct EmptyBase : Empty {
  double a, b;
};
void arg_empty_base(EmptyBase s) {}
// CHECK: define{{.*}} void @arg_empty_base([2 x i64] %{{.*}})

struct IndirectEmptyBase : EmptyBase {};
void arg_indirect_empty_base(IndirectEmptyBase s) {}
// CHECK: define{{.*}} void @arg_indirect_empty_base([2 x i64] %{{.*}})

// The rules also apply to the record type of each field, including array
// elements and union members.
struct PodField {
  Pod p;
};
void arg_pod_field(PodField s) {}
// CHECK: define{{.*}} void @arg_pod_field([2 x double] %{{.*}})

struct EmptyBaseField {
  EmptyBase e;
};
void arg_empty_base_field(EmptyBaseField s) {}
// CHECK: define{{.*}} void @arg_empty_base_field([2 x i64] %{{.*}})

struct EmptyBaseArrayField {
  EmptyBase e[1];
};
void arg_empty_base_array_field(EmptyBaseArrayField s) {}
// CHECK: define{{.*}} void @arg_empty_base_array_field([2 x i64] %{{.*}})

union EmptyBaseUnionMember {
  EmptyBase e;
};
void arg_empty_base_union_member(EmptyBaseUnionMember s) {}
// CHECK: define{{.*}} void @arg_empty_base_union_member([2 x i64] %{{.*}})

// Fields are not checked against the MSVC rules. An empty field that occupies
// no storage adds no padding, so the record is an HFA.
struct NoUniqueAddressEmpty {
  [[msvc::no_unique_address]] Empty e;
  double a, b;
};
void arg_no_unique_address_empty(NoUniqueAddressEmpty s) {}
// CHECK: define{{.*}} void @arg_no_unique_address_empty([2 x double] %{{.*}})

// A record whose only member is a zero-length array is not empty in the
// std::is_empty sense, so it does not disqualify a derived record.
struct ZeroLengthBase : ZeroLength {
  double a, b;
};
void arg_zero_length_base(ZeroLengthBase s) {}
// CHECK: define{{.*}} void @arg_zero_length_base([2 x double] %{{.*}})

union DoubleUnion {
  double a[2];
};
void arg_double_union(DoubleUnion s) {}
// CHECK: define{{.*}} void @arg_double_union([2 x double] %{{.*}})

union NonTrivialDefaultCtorUnion {
  double a[2] = {1.0, 2.0};
};
void arg_non_trivial_default_ctor_union(NonTrivialDefaultCtorUnion s) {}
// CHECK: define{{.*}} void @arg_non_trivial_default_ctor_union([2 x i64] %{{.*}})

}
