// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

template <typename T>
struct HasStaticInit {
static const int index;
};
extern "C"
int the_count = 0;
template <typename T>
const int HasStaticInit<T>::index = the_count++;

template <typename T> int func_tmpl1() { return HasStaticInit<T>::index; }
template <typename T> int func_tmpl2() { return HasStaticInit<T>::index; }
template <typename T> int func_tmpl3() { return HasStaticInit<T>::index; }
void useit() {
  func_tmpl1<int>();
  func_tmpl2<int>();
  func_tmpl3<int>();
}

// Throw in a final explicit instantiation to see that it doesn't screw things
// up.
template struct HasStaticInit<int>;

// There should only be one entry, not 3.
// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]

// There should only be one update to @the_count.
// CHECK-NOT: store i32 %{{.*}}, ptr @the_count
// CHECK: store i32 %{{.*}}, ptr @the_count
// CHECK-NOT: store i32 %{{.*}}, ptr @the_count
