// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: {\
// RUN:   misc-const-correctness.AnalyzeValues: false,\
// RUN:   misc-const-correctness.AnalyzeReferences: false,\
// RUN:   misc-const-correctness.AnalyzePointers: true,\
// RUN:   misc-const-correctness.WarnPointersAsValues: false,\
// RUN:   misc-const-correctness.WarnPointersAsPointers: true,\
// RUN:   misc-const-correctness.TransformPointersAsValues: false,\
// RUN:   misc-const-correctness.TransformPointersAsPointers: true\
// RUN: }}' \
// RUN: -- -fno-delayed-template-parsing

void pointee_to_const() {
  int a[] = {1, 2};
  int *p_local0 = &a[0];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p_local0' of type 'int *' can be declared 'const'
  // CHECK-FIXES: int  const*p_local0 = &a[0];
  p_local0 = &a[1];
}

void array_of_pointer_to_const() {
  int a[] = {1, 2};
  int *p_local0[1] = {&a[0]};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p_local0' of type 'int *[1]' can be declared 'const'
  // CHECK-FIXES: int  const*p_local0[1] = {&a[0]};
  p_local0[0] = &a[1];
}

template<class T>
void template_fn() {
  T a[] = {1, 2};
  T *p_local0 = &a[0];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: pointee of variable 'p_local0' of type 'char *' can be declared 'const'
  // CHECK-FIXES: T  const*p_local0 = &a[0];
  p_local0 = &a[1];
}

void instantiate() {
  template_fn<char>();
  template_fn<int>();
  template_fn<char const>();
}

using const_int = int const;
void ignore_const_alias() {
  const_int a[] = {1, 2};
  const_int *p_local0 = &a[0];
  p_local0 = &a[1];
}

void function_pointer_basic() {
  void (*const fp)() = nullptr;
  fp();
}

void takeNonConstRef(int *&r);

void ignoreNonConstRefOps() {
  // init with non-const ref
  int* p0 {nullptr};
  int*& r1 = p0;
  
  // non-const ref param
  int* p1 {nullptr};
  takeNonConstRef(p1);

  // cast
  int* p2 {nullptr};
  int*& r2 = (int*&)p2;
}
