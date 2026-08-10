// RUN: %clang_cc1 %s -fenable-matrix -pedantic -std=c++11 -verify -triple=x86_64-apple-darwin9

template <typename EltTy, unsigned Rows, unsigned Columns>
struct MyMatrix {
  using matrix_t = EltTy __attribute__((matrix_type(Rows, Columns)));

  matrix_t value;
};

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1>
typename MyMatrix<EltTy1, R1, C1>::matrix_t transpose(MyMatrix<EltTy0, R0, C0> &A) {
  char *v1 = __builtin_matrix_transpose(A.value);
  // expected-error@-1 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 2)))'}}
  // expected-error@-2 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}
  // expected-error@-3 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}

  __builtin_matrix_transpose(A);
  // expected-error@-1 {{1st argument must be a matrix}}
  // expected-error@-2 {{1st argument must be a matrix}}
  // expected-error@-3 {{1st argument must be a matrix}}

  return __builtin_matrix_transpose(A.value);
  // expected-error@-1 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 2U, 3U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 3)))') with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 2)))'}}
  // expected-error@-2 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 2U, 3U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 3)))') with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}
  // expected-error@-3 {{cannot initialize return object of type 'typename MyMatrix<float, 3U, 3U>::matrix_t' (aka 'float __attribute__((matrix_type(3, 3)))') with an rvalue of type 'unsigned int __attribute__((matrix_type(3, 3)))'}}
}

void test_transpose_template(unsigned *Ptr1, float *Ptr2) {
  MyMatrix<unsigned, 2, 3> Mat1;
  MyMatrix<unsigned, 3, 3> Mat2;
  Mat1.value = *((decltype(Mat1)::matrix_t *)Ptr1);
  Mat1.value = transpose<unsigned, 2, 3, unsigned, 2, 3>(Mat1);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 2U, 3U, unsigned int, 2U, 3U>' requested here}}

  Mat1.value = transpose<unsigned, 3, 3, unsigned, 2, 3>(Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 3U, 3U, unsigned int, 2U, 3U>' requested here}}

  MyMatrix<float, 3, 3> Mat3;
  Mat3.value = transpose<unsigned, 3, 3, float, 3, 3>(Mat2);
  // expected-note@-1 {{in instantiation of function template specialization 'transpose<unsigned int, 3U, 3U, float, 3U, 3U>' requested here}}
}

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1, unsigned R1, unsigned C1>
typename MyMatrix<EltTy1, R1, C1>::matrix_t column_major_load(MyMatrix<EltTy0, R0, C0> &A, EltTy0 *Ptr) {
  char *v1 = __builtin_matrix_column_major_load(Ptr, 9, 4, 10);
  // expected-error@-1 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(9, 4)))'}}
  // expected-error@-2 {{cannot initialize a variable of type 'char *' with an rvalue of type 'unsigned int __attribute__((matrix_type(9, 4)))'}}
  // expected-error@-3 {{cannot initialize a variable of type 'char *' with an rvalue of type 'float __attribute__((matrix_type(9, 4)))'}}

  return __builtin_matrix_column_major_load(Ptr, R0, C0, R0);
  // expected-error@-1 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 5U, 5U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(5, 5)))') with an rvalue of type 'unsigned int __attribute__((matrix_type(2, 3)))'}}
  // expected-error@-2 {{cannot initialize return object of type 'typename MyMatrix<unsigned int, 2U, 3U>::matrix_t' (aka 'unsigned int __attribute__((matrix_type(2, 3)))') with an rvalue of type 'float __attribute__((matrix_type(2, 3)))'}}
}

void test_column_major_loads_template(unsigned *Ptr1, float *Ptr2) {
  MyMatrix<unsigned, 2, 3> Mat1;
  Mat1.value = column_major_load<unsigned, 2, 3, unsigned, 2, 3>(Mat1, Ptr1);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_load<unsigned int, 2U, 3U, unsigned int, 2U, 3U>' requested here}}
  column_major_load<unsigned, 2, 3, unsigned, 5, 5>(Mat1, Ptr1);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_load<unsigned int, 2U, 3U, unsigned int, 5U, 5U>' requested here}}

  MyMatrix<float, 2, 3> Mat2;
  Mat1.value = column_major_load<float, 2, 3, unsigned, 2, 3>(Mat2, Ptr2);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_load<float, 2U, 3U, unsigned int, 2U, 3U>' requested here}}
}

constexpr int constexpr1() { return 1; }
constexpr int constexpr_neg1() { return -1; }

void test_column_major_load_constexpr(unsigned *Ptr) {
  (void)__builtin_matrix_column_major_load(Ptr, 2, 2, constexpr1());
  // expected-error@-1 {{stride must be greater or equal to the number of rows}}
  (void)__builtin_matrix_column_major_load(Ptr, constexpr_neg1(), 2, 4);
  // expected-error@-1 {{row dimension is outside the allowed range [1, 1048575]}}
  (void)__builtin_matrix_column_major_load(Ptr, 2, constexpr_neg1(), 4);
  // expected-error@-1 {{column dimension is outside the allowed range [1, 1048575]}}
}

struct IntWrapper {
  operator int() {
    return 1;
  }
};

void test_column_major_load_wrapper(unsigned *Ptr, IntWrapper &W) {
  (void)__builtin_matrix_column_major_load(Ptr, W, 2, 2);
  // expected-error@-1 {{row argument must be a constant unsigned integer expression}}
  (void)__builtin_matrix_column_major_load(Ptr, 2, W, 2);
  // expected-error@-1 {{column argument must be a constant unsigned integer expression}}
}

template <typename T, unsigned R, unsigned C, unsigned S>
void test_column_major_load_temp(T Ptr) {
  (void)__builtin_matrix_column_major_load(Ptr, R, C, S);
}

void call_column_major_load_temp(unsigned *Ptr, unsigned X) {
  (void)__builtin_matrix_column_major_load(Ptr, X, X, X);
  // expected-error@-1 {{row argument must be a constant unsigned integer expression}}
  // expected-error@-2 {{column argument must be a constant unsigned integer expression}}
  (void)__builtin_matrix_column_major_load(X, 2, 2, 2);
  // expected-error@-1 {{1st argument must be a pointer to a valid matrix element type}}
}

template <typename EltTy0, unsigned R0, unsigned C0, typename PtrTy>
void column_major_store(MyMatrix<EltTy0, R0, C0> &A, PtrTy Ptr, unsigned Stride) {
  __builtin_matrix_column_major_store(A.value, Ptr, Stride);
  // expected-error@-1 {{the pointee of the 2nd argument must match the element type of the 1st argument ('float' != 'unsigned int')}}
}

template <typename MTy, typename PtrTy, unsigned Stride>
void column_major_store(MTy &A, PtrTy Ptr) {
  __builtin_matrix_column_major_store(A.value, Ptr, Stride);
  // expected-error@-1 {{stride must be greater or equal to the number of rows}}
}

void test_column_major_stores_template(MyMatrix<unsigned, 2, 3> &M1, unsigned *Ptr1, MyMatrix<float, 3, 4> &M2, float *Ptr2) {
  column_major_store(M1, Ptr2, 10);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_store<unsigned int, 2U, 3U, float *>' requested here}}

  column_major_store<decltype(M2), float *, 1>(M2, Ptr2);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_store<MyMatrix<float, 3, 4> &, float *, 1U>' requested here}}
}

template <typename EltTy0, unsigned R0, unsigned C0, typename EltTy1>
void column_major_store(MyMatrix<EltTy0, R0, C0> &A, EltTy1 *Ptr) {
  __builtin_matrix_column_major_store(A.value, Ptr, 1);
  // expected-error@-1 3 {{stride must be greater or equal to the number of rows}}
  // expected-error@-2 {{the pointee of the 2nd argument must match the element type of the 1st argument ('float' != 'unsigned int')}}
  // expected-error@-3 {{the pointee of the 2nd argument must match the element type of the 1st argument ('unsigned int' != 'float')}}

  char *s;
  return __builtin_matrix_column_major_store(A.value, s, 20);
  // expected-error@-1 {{the pointee of the 2nd argument must match the element type of the 1st argument ('char' != 'unsigned int')}}
  // expected-error@-2 {{the pointee of the 2nd argument must match the element type of the 1st argument ('char' != 'unsigned int')}}
  // expected-error@-3 {{he pointee of the 2nd argument must match the element type of the 1st argument ('char' != 'float')}}
}

void test_column_major_store_template(unsigned *Ptr1, float *Ptr2) {
  MyMatrix<unsigned, 2, 3> Mat1;
  column_major_store<unsigned, 2, 3, unsigned>(Mat1, Ptr1);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_store<unsigned int, 2U, 3U, unsigned int>'}}
  column_major_store<unsigned, 2, 3, float>(Mat1, Ptr2);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_store<unsigned int, 2U, 3U, float>'}}

  MyMatrix<float, 2, 3> Mat2;
  column_major_store<float, 2, 3, unsigned>(Mat2, Ptr1);
  // expected-note@-1 {{in instantiation of function template specialization 'column_major_store<float, 2U, 3U, unsigned int>'}}
}

void test_column_major_store_constexpr(unsigned *Ptr, MyMatrix<unsigned, 3, 3> &M) {
  __builtin_matrix_column_major_store(M.value, Ptr, constexpr1());
  // expected-error@-1 {{stride must be greater or equal to the number of rows}}
  __builtin_matrix_column_major_store(constexpr1(), Ptr, 1);
  // expected-error@-1 {{1st argument must be a matrix}}
  __builtin_matrix_column_major_store(M.value, constexpr1(), 1);
  // expected-error@-1 {{2nd argument must be a pointer to a valid matrix element type}}
  // expected-error@-2 {{stride must be greater or equal to the number of rows}}
}

void test_column_major_store_wrapper(unsigned *Ptr, MyMatrix<unsigned, 3, 3> &M, IntWrapper &W) {
  __builtin_matrix_column_major_store(M.value, Ptr, W);

  __builtin_matrix_column_major_store(W, Ptr, W);
  // expected-error@-1 {{1st argument must be a matrix}}
}
