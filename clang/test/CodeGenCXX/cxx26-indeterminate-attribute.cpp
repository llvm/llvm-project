// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s -check-prefix=CXX26
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO
// RUN: %clang_cc1 -std=c++26 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN

// Test for C++26 [[indeterminate]] attribute (P2795R5)
// The [[indeterminate]] attribute opts out of erroneous initialization,
// suppressing trivial auto var init even with -ftrivial-auto-var-init.

template<typename T> void used(T &) noexcept;

extern "C" {

// [[indeterminate]] suppresses all initialization: no store between alloca and use.
// CXX26-LABEL:   @test_indeterminate_local_var(
// CXX26:         %x = alloca i32
// CXX26-NOT:     store
// CXX26:         call void @_Z4usedIiEvRT_(
// ZERO-LABEL:    @test_indeterminate_local_var(
// ZERO:          %x = alloca i32
// ZERO-NOT:      store
// ZERO:          call void @_Z4usedIiEvRT_(
// PATTERN-LABEL: @test_indeterminate_local_var(
// PATTERN:       %x = alloca i32
// PATTERN-NOT:   store
// PATTERN:       call void @_Z4usedIiEvRT_(
void test_indeterminate_local_var() {
  [[indeterminate]] int x;
  used(x);
}

// Without [[indeterminate]], C++26 erroneous initialization applies:
// the alloca is filled with a fixed pattern so reads have a well-defined
// (if implementation-defined) value, rather than LLVM `undef`.
// CXX26-LABEL:   @test_normal_local_var(
// CXX26:         %y = alloca i32
// CXX26:         store i32 -1431655766, ptr %y
// CXX26:         call void @_Z4usedIiEvRT_(
// ZERO-LABEL:    @test_normal_local_var(
// ZERO:          %y = alloca i32
// ZERO:          store i32 0, ptr %y
// ZERO:          call void @_Z4usedIiEvRT_(
// PATTERN-LABEL: @test_normal_local_var(
// PATTERN:       %y = alloca i32
// PATTERN:       store i32 -1431655766, ptr %y
// PATTERN:       call void @_Z4usedIiEvRT_(
void test_normal_local_var() {
  int y;
  used(y);
}

// [[indeterminate]] on multiple variables: no memset or store for any of them.
// ZERO-LABEL:    @test_indeterminate_multiple_vars(
// ZERO:          %a = alloca i32
// ZERO:          %b = alloca [10 x i32]
// ZERO:          %c = alloca [10 x [10 x i32]]
// ZERO-NOT:      store
// ZERO-NOT:      call void @llvm.memset
// ZERO:          call void @_Z4usedIiEvRT_(
void test_indeterminate_multiple_vars() {
  [[indeterminate]] int a, b[10], c[10][10];
  used(a);
}

// Mixed: normal var is zero-initialized by its empty-brace init,
// [[indeterminate]] var gets no fill, erroneous var (no attribute) gets
// the C++26 erroneous-init fill (pattern by default, zero under
// -ftrivial-auto-var-init=zero).
// CXX26-LABEL:   @test_mixed_vars(
// CXX26:         %normal = alloca i32
// CXX26:         %indeterminate_var = alloca i32
// CXX26:         %erroneous = alloca i32
// CXX26:         store i32 0, ptr %normal
// CXX26-NOT:     store {{.*}} ptr %indeterminate_var
// CXX26:         store i32 -1431655766, ptr %erroneous
// CXX26:         call void @_Z4usedIiEvRT_(ptr {{.*}} %normal)
// CXX26:         call void @_Z4usedIiEvRT_(ptr {{.*}} %indeterminate_var)
// CXX26:         call void @_Z4usedIiEvRT_(ptr {{.*}} %erroneous)
// ZERO-LABEL:    @test_mixed_vars(
// ZERO:          %normal = alloca i32
// ZERO:          %indeterminate_var = alloca i32
// ZERO:          %erroneous = alloca i32
// ZERO:          store i32 0, ptr %normal
// ZERO-NOT:      store {{.*}} ptr %indeterminate_var
// ZERO:          store i32 0, ptr %erroneous
// ZERO:          call void @_Z4usedIiEvRT_(ptr {{.*}} %normal)
// ZERO:          call void @_Z4usedIiEvRT_(ptr {{.*}} %indeterminate_var)
// ZERO:          call void @_Z4usedIiEvRT_(ptr {{.*}} %erroneous)
// PATTERN-LABEL: @test_mixed_vars(
// PATTERN:       %normal = alloca i32
// PATTERN:       %indeterminate_var = alloca i32
// PATTERN:       %erroneous = alloca i32
// PATTERN:       store i32 0, ptr %normal
// PATTERN-NOT:   store {{.*}} ptr %indeterminate_var
// PATTERN:       store i32 -1431655766, ptr %erroneous
// PATTERN:       call void @_Z4usedIiEvRT_(ptr {{.*}} %normal)
// PATTERN:       call void @_Z4usedIiEvRT_(ptr {{.*}} %indeterminate_var)
// PATTERN:       call void @_Z4usedIiEvRT_(ptr {{.*}} %erroneous)
void test_mixed_vars() {
  int normal = {};               // Explicitly zero-initialized
  [[indeterminate]] int indeterminate_var;
  int erroneous;                 // Will get zero-init with -ftrivial-auto-var-init=zero
  used(normal);
  used(indeterminate_var);
  used(erroneous);
}

} // extern "C"

// Test: Struct with indeterminate member initialization
struct SelfStorage {
  char data[512];
  void use_data();
};

// ZERO-LABEL:    @_Z25test_struct_indeterminatev(
// ZERO:          %s = alloca %struct.SelfStorage
// ZERO-NOT:      call void @llvm.memset
// ZERO:          call void @_ZN11SelfStorage8use_dataEv(
void test_struct_indeterminate() {
  [[indeterminate]] SelfStorage s;
  s.use_data();
}

// Without [[indeterminate]], struct gets zero-init.
// ZERO-LABEL:    @_Z18test_struct_normalv(
// ZERO:          %s = alloca %struct.SelfStorage
// ZERO:          call void @llvm.memset
// ZERO:          call void @_ZN11SelfStorage8use_dataEv(
void test_struct_normal() {
  SelfStorage s;
  s.use_data();
}
