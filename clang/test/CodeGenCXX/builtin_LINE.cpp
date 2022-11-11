// RUN: %clang_cc1 -std=c++1z -fblocks %s -triple x86_64-unknown-unknown -emit-llvm -o - | FileCheck %s

extern "C" int sink;
extern "C" const volatile void* volatile ptr_sink = nullptr;

struct Tag1 {};
struct Tag2 {};
struct Tag3 {};
struct Tag4 {};

constexpr int get_line_constexpr(int l = __builtin_LINE()) {
  return l;
}

int get_line_nonconstexpr(int l = __builtin_LINE()) {
  return l;
}


int get_line(int l = __builtin_LINE()) {
  return l;
}

int get_line2(int l = get_line()) { return l; }


// CHECK: @global_one ={{.*}} global i32 [[@LINE+1]], align 4
int global_one = __builtin_LINE();
// CHECK-NEXT: @global_two ={{.*}} global i32 [[@LINE+1]], align 4
int global_two = get_line_constexpr();
// CHECK: @_ZL12global_three = internal constant i32 [[@LINE+1]], align 4
const int global_three(get_line_constexpr());

// CHECK-LABEL: define internal void @__cxx_global_var_init
// CHECK: %call = call noundef i32 @_Z21get_line_nonconstexpri(i32 noundef [[@LINE+2]])
// CHECK-NEXT: store i32 %call, ptr @global_four, align 4
int global_four = get_line_nonconstexpr();

struct InClassInit {
  int Init = __builtin_LINE();
  int Init2 = get_line2();
  InClassInit();
  constexpr InClassInit(Tag1, int l = __builtin_LINE()) : Init(l), Init2(l) {}
  constexpr InClassInit(Tag2) : Init(__builtin_LINE()), Init2(__builtin_LINE()) {}
  InClassInit(Tag3, int l = __builtin_LINE());
  InClassInit(Tag4, int l = get_line2());

  static void test_class();
};
// CHECK-LABEL: define{{.*}} void @_ZN11InClassInit10test_classEv()
void InClassInit::test_class() {
  // CHECK: call void @_ZN11InClassInitC1Ev(ptr {{[^,]*}} %test_one)
  InClassInit test_one;
  // CHECK-NEXT: call void @_ZN11InClassInitC1E4Tag1i(ptr {{[^,]*}} %test_two, i32 noundef [[@LINE+1]])
  InClassInit test_two{Tag1{}};
  // CHECK-NEXT: call void @_ZN11InClassInitC1E4Tag2(ptr {{[^,]*}} %test_three)
  InClassInit test_three{Tag2{}};
  // CHECK-NEXT: call void @_ZN11InClassInitC1E4Tag3i(ptr {{[^,]*}} %test_four, i32 noundef [[@LINE+1]])
  InClassInit test_four(Tag3{});
  // CHECK-NEXT: %[[CALL:.+]] = call noundef i32 @_Z8get_linei(i32 noundef [[@LINE+3]])
  // CHECK-NEXT: %[[CALL2:.+]] = call noundef i32 @_Z9get_line2i(i32 noundef %[[CALL]])
  // CHECK-NEXT: call void @_ZN11InClassInitC1E4Tag4i(ptr {{[^,]*}} %test_five, i32 noundef %[[CALL2]])
  InClassInit test_five(Tag4{});

}
// CHECK-LABEL: define{{.*}} void @_ZN11InClassInitC2Ev
// CHECK: store i32 [[@LINE+4]], ptr %Init, align 4
// CHECK: %call = call noundef i32 @_Z8get_linei(i32 noundef [[@LINE+3]])
// CHECK-NEXT: %call2 = call noundef i32 @_Z9get_line2i(i32 noundef %call)
// CHECK-NEXT: store i32 %call2, ptr %Init2, align 4
InClassInit::InClassInit() = default;

InClassInit::InClassInit(Tag3, int l) : Init(l) {}

// CHECK-LABEL: define{{.*}} void @_ZN11InClassInitC2E4Tag4i(ptr {{[^,]*}} %this, i32 noundef %arg)
// CHECK:  %[[TEMP:.+]] = load i32, ptr %arg.addr, align 4
// CHECK-NEXT: store i32 %[[TEMP]], ptr %Init, align 4
// CHECK: %[[CALL:.+]] = call noundef i32 @_Z8get_linei(i32 noundef [[@LINE+3]])
// CHECK-NEXT: %[[CALL2:.+]] = call noundef i32 @_Z9get_line2i(i32 noundef %[[CALL]])
// CHECK-NEXT: store i32 %[[CALL2]], ptr %Init2, align 4
InClassInit::InClassInit(Tag4, int arg) : Init(arg) {}

// CHECK-LABEL: define{{.*}} void @_Z13get_line_testv()
void get_line_test() {
  // CHECK: %[[CALL:.+]] = call noundef i32 @_Z8get_linei(i32 noundef [[@LINE+2]])
  // CHECK-NEXT: store i32 %[[CALL]], ptr @sink, align 4
  sink = get_line();
  // CHECK-NEXT:  store i32 [[@LINE+1]], ptr @sink, align 4
  sink = __builtin_LINE();
  ptr_sink = &global_three;
}

void foo() {
  const int N[] = {__builtin_LINE(), get_line_constexpr()};
}
