// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -std=c++11 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// Test that iterator loop variables are correctly finalized after loop-transformation
// constructs as required by OpenMP 6.0 spec (pg 371, lines 19-21).

// Simple iterator class for testing
struct Iterator {
  int *ptr;

  Iterator(int *p) : ptr(p) {}

  bool operator!=(const Iterator &other) const {
    return ptr != other.ptr;
  }

  Iterator &operator++() {
    ++ptr;
    return *this;
  }

  int &operator*() {
    return *ptr;
  }

  long operator-(const Iterator &other) const {
    return ptr - other.ptr;
  }

  Iterator operator+(long n) const {
    return Iterator(ptr + n);
  }
};

struct Container {
  int data[10];

  Iterator begin() { return Iterator(data); }
  Iterator end() { return Iterator(data + 10); }
};

void test_iterator_tile(void) {
  // CHECK-LABEL: define {{.*}} @_Z{{.*}}test_iterator_tile
  Container cont;
  Iterator it = cont.begin();
  #pragma omp tile sizes(3)
  for (it = cont.begin(); it != cont.end(); ++it) {
    *it = 42;
  }
  // CHECK: for.end{{.*}}:
  // Iterator should be finalized to cont.end() (loop-exit value)
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %it
}

void test_iterator_reverse(void) {
  // CHECK-LABEL: define {{.*}} @_Z{{.*}}test_iterator_reverse
  Container cont;
  Iterator it = cont.begin();
  #pragma omp reverse
  for (it = cont.begin(); it != cont.end(); ++it) {
    *it = 42;
  }
  // CHECK: for.end{{.*}}:
  // Iterator should be finalized to cont.end() (loop-exit value)
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %it
}

struct Container2 {
  int data[5];

  Iterator begin() { return Iterator(data); }
  Iterator end() { return Iterator(data + 5); }
};

void test_iterator_interchange(void) {
  // CHECK-LABEL: define {{.*}} @_Z{{.*}}test_iterator_interchange
  Container cont1;
  Container2 cont2;
  Iterator i = cont1.begin();
  Iterator j = cont2.begin();
  #pragma omp interchange permutation(2, 1)
  for (i = cont1.begin(); i != cont1.end(); ++i) {
    for (j = cont2.begin(); j != cont2.end(); ++j) {
      // noop
    }
  }
  // CHECK: for.end{{.*}}:
  // Both iterators should be finalized to their respective end() values
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %i
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %j
}

void test_iterator_fuse(void) {
  // CHECK-LABEL: define {{.*}} @_Z{{.*}}test_iterator_fuse
  Container cont1;
  Container2 cont2;
  Iterator p = cont1.begin();
  Iterator q = cont2.begin();
  #pragma omp fuse
  {
    for (p = cont1.begin(); p != cont1.end(); ++p) {
      *p = 1;
    }
    for (q = cont2.begin(); q != cont2.end(); ++q) {
      *q = 2;
    }
  }
  // CHECK: for.end{{.*}}:
  // Both iterators should be finalized to their respective end() values
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %p
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %q
}

void test_iterator_tile_nested(void) {
  // CHECK-LABEL: define {{.*}} @_Z{{.*}}test_iterator_tile_nested
  Container cont1;
  Container2 cont2;
  Iterator i = cont1.begin();
  Iterator j = cont2.begin();
  #pragma omp tile sizes(2, 3)
  for (i = cont1.begin(); i != cont1.end(); ++i) {
    for (j = cont2.begin(); j != cont2.end(); ++j) {
      // noop
    }
  }
  // CHECK: for.end{{.*}}:
  // Both iterators should be finalized to their respective end() values
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %i
  // CHECK: call {{.*}} @_ZNK8IteratormiERKS_
  // CHECK: call {{.*}} @_ZNK8IteratorplEl
  // CHECK: call void @llvm.memcpy{{.*}}(ptr align {{.*}} %j
}
