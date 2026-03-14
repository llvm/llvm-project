// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s

// Not trivially copyable because of the explicit destructor.
// CHECK-DAG: !DICompositeType({{.*}}, name: "RefDtor",{{.*}}flags: DIFlagTypePassByReference
struct RefDtor {
  int i;
  ~RefDtor() {}
} refDtor;

// Not trivially copyable because of the explicit copy constructor.
// CHECK-DAG: !DICompositeType({{.*}}, name: "RefCopy",{{.*}}flags: DIFlagTypePassByReference
struct RefCopy {
  int i;
  RefCopy() = default;
  RefCopy(RefCopy &Copy) {}
} refCopy;

// POD-like type even though it defines a destructor.
// CHECK-DAG: !DICompositeType({{.*}}, name: "Podlike", {{.*}}flags: DIFlagTypePassByValue
struct Podlike {
  int i;
  Podlike() = default;
  Podlike(Podlike &&Move) = default;
  ~Podlike() = default;
} podlike;


// This is a POD type.
// CHECK-DAG: !DICompositeType({{.*}}, name: "Pod",{{.*}}flags: DIFlagTypePassByValue
struct Pod {
  int i;
} pod;

// This is definitely not a POD type.
// CHECK-DAG: !DICompositeType({{.*}}, name: "Complex",{{.*}}flags: DIFlagTypePassByReference
struct Complex {
  Complex() {}
  Complex(Complex &Copy) : i(Copy.i) {};
  int i;
} complex;

// This type is manually marked as trivial_abi.
// CHECK-DAG: !DICompositeType({{.*}}, name: "Marked",{{.*}}flags: DIFlagTypePassByValue
struct __attribute__((trivial_abi)) Marked {
  int *p;
  Marked();
  ~Marked();
  Marked(const Marked &) noexcept;
  Marked &operator=(const Marked &);
} marked;
