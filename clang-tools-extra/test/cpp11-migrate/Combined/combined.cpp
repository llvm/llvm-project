// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t_risky.cpp
// RUN: cpp11-migrate -loop-convert -use-nullptr %t.cpp --
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: cpp11-migrate -loop-convert -use-nullptr -risk=risky %t_risky.cpp --
// RUN: FileCheck -check-prefix=RISKY -input-file=%t_risky.cpp %s

#define NULL 0

struct T {
  struct iterator {
    int *& operator*();
    const int *& operator*() const;
    iterator & operator++();
    bool operator!=(const iterator &other);
    void insert(int *);
    int *x;
  };

  iterator begin();
  iterator end();
};

void test_loopconvert_and_nullptr_iterator() {
  T t;

  for (T::iterator it = t.begin(); it != t.end(); ++it) {
    *it = NULL;
  }

  // CHECK: for (auto & elem : t)
  // CHECK-NEXT: elem = nullptr;
}

void test_loopconvert_and_nullptr_risky() {
  const int N = 10;
  int *(*pArr)[N];

  for (int i = 0; i < N; ++i) {
    (*pArr)[i] = NULL;
  }

  // RISKY: for (auto & elem : *pArr)
  // RISKY-NEXT: elem = nullptr;
}
