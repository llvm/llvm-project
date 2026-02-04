// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: \
// RUN:  {misc-const-correctness.AnalyzeValues: true,\
// RUN:   misc-const-correctness.WarnPointersAsValues: true,\
// RUN:   misc-const-correctness.WarnPointersAsPointers: false,\
// RUN:   misc-const-correctness.TransformPointersAsValues: true}}' \
// RUN: -- -fno-delayed-template-parsing

void potential_const_pointer() {
  double np_local0[10] = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
  double *p_local0 = &np_local0[1];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'double *' can be declared 'const'
  // CHECK-FIXES: double *const p_local0 = &np_local0[1];

  using doublePtr = double*;
  using doubleArray = double[15];
  doubleArray np_local1;
  doublePtr p_local1 = &np_local1[0];
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local1' of type 'doublePtr' (aka 'double *') can be declared 'const'
  // CHECK-FIXES: doublePtr const p_local1 = &np_local1[0];
}

void range_for() {
  int np_local0[2] = {1, 2};
  int *p_local0[2] = {&np_local0[0], &np_local0[1]};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int *[2]' can be declared 'const'
  // CHECK-FIXES: int *const p_local0[2] = {&np_local0[0], &np_local0[1]};
  for (const int *p_local1 : p_local0) {
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: variable 'p_local1' of type 'const int *' can be declared 'const'
  // CHECK-FIXES: for (const int *const p_local1 : p_local0) {
  }

  int *p_local2[2] = {nullptr, nullptr};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local2' of type 'int *[2]' can be declared 'const'
  // CHECK-FIXES: int *const p_local2[2] = {nullptr, nullptr};
  for (const auto *con_ptr : p_local2) {
  }
}

template <typename T>
struct SmallVectorBase {
  T data[4];
  void push_back(const T &el) {}
  int size() const { return 4; }
  T *begin() { return data; }
  const T *begin() const { return data; }
  T *end() { return data + 4; }
  const T *end() const { return data + 4; }
};

template <typename T>
struct SmallVector : SmallVectorBase<T> {};

template <class T>
void EmitProtocolMethodList(T &&Methods) {
  // Note: If the template is uninstantiated the analysis does not figure out,
  // that p_local0 could be const. Not sure why, but probably bails because
  // some expressions are type-dependent.
  SmallVector<const int *> p_local0;
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'SmallVector<const int *>' can be declared 'const'
  // CHECK-FIXES: SmallVector<const int *> const p_local0;
  SmallVector<const int *> np_local0;
  for (const auto *I : Methods) {
    if (I == nullptr)
      np_local0.push_back(I);
  }
  p_local0.size();
}
void instantiate() {
  int *p_local0[4] = {nullptr, nullptr, nullptr, nullptr};
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'p_local0' of type 'int *[4]' can be declared 'const'
  // CHECK-FIXES: int *const p_local0[4] = {nullptr, nullptr, nullptr, nullptr};
  EmitProtocolMethodList(p_local0);
}
