// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CWG 3043: The temporary bound to f's parameter lives as long as the
// expansion variable, so it must be destroyed after the body.

struct T {
  int& x;
  T(int& x) noexcept : x(x) {}
  ~T() noexcept { x = 42; }
};

const T& f(const T& t) noexcept { return t; }
T g(int& x) noexcept { return T(x); }
void body(int);

int lifetime_extension() {
  int x = 5;
  template for (auto&& e : {f(g(x)), f(g(x))}) {
    body(e.x);
  }
  return x;
}

template <typename U>
int lifetime_extension_instantiate_expansions() {
  int x = 5;
  template for (U e : {f(g(x))}) {
    body(e.x);
  }
  return x;
}

template <typename... Ts>
int lifetime_extension_pack(Ts... ts) {
  int x = 5;
  template for (auto&& e : {f(g(x)), f(g(ts))...}) {
    body(e.x);
  }
  return x;
}

void instantiate() {
  lifetime_extension_instantiate_expansions<const T&>();
  lifetime_extension_pack(1);
}

// CHECK-LABEL: define {{.*}} i32 @_Z18lifetime_extensionv()
// CHECK:       call void @_Z1gRi(ptr {{.*}}sret{{.*}} %[[TMP0:[^ ,]+]], ptr {{.*}} %x)
// CHECK-NEXT:  call {{.*}} ptr @_Z1fRK1T(ptr {{.*}} %[[TMP0]])
// CHECK-NOT:   call void @_ZN1TD1Ev
// CHECK:       call void @_Z4bodyi(
// CHECK:       call void @_ZN1TD1Ev(ptr {{.*}} %[[TMP0]])
// CHECK:       call void @_Z1gRi(ptr {{.*}}sret{{.*}} %[[TMP1:[^ ,]+]], ptr {{.*}} %x)
// CHECK-NEXT:  call {{.*}} ptr @_Z1fRK1T(ptr {{.*}} %[[TMP1]])
// CHECK-NOT:   call void @_ZN1TD1Ev
// CHECK:       call void @_Z4bodyi(
// CHECK:       call void @_ZN1TD1Ev(ptr {{.*}} %[[TMP1]])
// CHECK:       ret i32

// CHECK-LABEL: define {{.*}} i32 @_Z41lifetime_extension_instantiate_expansionsIRK1TEiv()
// CHECK:       call void @_Z1gRi(ptr {{.*}}sret{{.*}} %[[TMP2:[^ ,]+]], ptr {{.*}} %x)
// CHECK-NEXT:  call {{.*}} ptr @_Z1fRK1T(ptr {{.*}} %[[TMP2]])
// CHECK-NOT:   call void @_ZN1TD1Ev
// CHECK:       call void @_Z4bodyi(
// CHECK:       call void @_ZN1TD1Ev(ptr {{.*}} %[[TMP2]])
// CHECK:       ret i32

// CHECK-LABEL: define {{.*}} i32 @_Z23lifetime_extension_packIJiEEiDpT_(
// CHECK:       call void @_Z1gRi(ptr {{.*}}sret{{.*}} %[[TMP3:[^ ,]+]], ptr {{.*}} %x)
// CHECK-NEXT:  call {{.*}} ptr @_Z1fRK1T(ptr {{.*}} %[[TMP3]])
// CHECK-NOT:   call void @_ZN1TD1Ev
// CHECK:       call void @_Z4bodyi(
// CHECK:       call void @_ZN1TD1Ev(ptr {{.*}} %[[TMP3]])
// CHECK:       call void @_Z1gRi(ptr {{.*}}sret{{.*}} %[[TMP4:[^ ,]+]], ptr {{.*}} %ts
// CHECK-NEXT:  call {{.*}} ptr @_Z1fRK1T(ptr {{.*}} %[[TMP4]])
// CHECK-NOT:   call void @_ZN1TD1Ev
// CHECK:       call void @_Z4bodyi(
// CHECK:       call void @_ZN1TD1Ev(ptr {{.*}} %[[TMP4]])
// CHECK:       ret i32
