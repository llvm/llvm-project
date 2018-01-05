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

// Not trivially copyable because of the explicit move constructor.
// CHECK-DAG: !DICompositeType({{.*}}, name: "RefMove",{{.*}}flags: DIFlagTypePassByReference
struct RefMove {
  int i;
  RefMove() = default;
  RefMove(RefMove &&Move) {}
} refMove;

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
