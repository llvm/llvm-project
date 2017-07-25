// RUN: %check_clang_tidy %s bugprone-undefined-memory-manipulation %t

void *memset(void *, int, __SIZE_TYPE__);
void *memcpy(void *, const void *, __SIZE_TYPE__);
void *memmove(void *, const void *, __SIZE_TYPE__);

namespace std {
using ::memcpy;
using ::memmove;
using ::memset;
}

// TriviallyCopyable types:
struct Plain {
  int n;
};

enum E {
  X,
  Y,
  Z
};

struct Base {
  float b;
};

struct Derived : Base {
  bool d;
};

// not TriviallyCopyable types:
struct Destruct {
  ~Destruct() {}
};

struct Copy {
  Copy() {}
  Copy(const Copy &) {}
};

struct Move {
  Move() {}
  Move(Move &&) {}
};

struct VirtualFunc {
  virtual void f() {}
};

struct VirtualBase : virtual Base {
  int vb;
};

// Incomplete type, assume it is TriviallyCopyable.
struct NoDef;

void f(NoDef *s) {
  memset(s, 0, 5);
}

template <typename T>
void memset_temp(T *b) {
  memset(b, 0, sizeof(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
}

template <typename S, typename T>
void memcpy_temp(S *a, T *b) {
  memcpy(a, b, sizeof(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
}

template <typename S, typename T>
void memmove_temp(S *a, T *b) {
  memmove(a, b, sizeof(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
}

void notTriviallyCopyable() {
  Plain p; // TriviallyCopyable for variety
  Destruct d;
  Copy c;
  Move m;
  VirtualFunc vf;
  VirtualBase vb;

  memset(&vf, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  memset(&d, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  memset(&c, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  std::memset(&m, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  ::memset(&vb, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]

  memcpy(&p, &vf, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  memcpy(&p, &d, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  memcpy(&c, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  std::memcpy(&m, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  ::memcpy(&vb, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]

  memmove(&vf, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  memmove(&d, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  memmove(&p, &c, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  std::memmove(&p, &m, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
  ::memmove(&p, &vb, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]

#define MEMSET memset(&vf, 0, sizeof(int));
  MEMSET
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
#define MEMCPY memcpy(&d, &p, sizeof(int));
  MEMCPY
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
#define MEMMOVE memmove(&p, &c, sizeof(int));
  MEMMOVE
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object is not TriviallyCopyable [bugprone-undefined-memory-manipulation]

  memset_temp<VirtualFunc>(&vf);
  memcpy_temp<Plain, VirtualFunc>(&p, &vf);
  memmove_temp<Plain, VirtualFunc>(&p, &vf);
}

void triviallyCopyable() {
  Plain p;
  Base base;
  Derived derived;

  int i = 5;
  int ia[3] = {1, 2, 3};
  float f = 3.14;
  float fa[3] = {1.1, 2.2, 3.3};
  bool b = false;
  bool ba[2] = {true, false};
  E e = X;
  p.n = 2;

  memset(&p, 0, sizeof(int));
  memset(&base, 0, sizeof(float));
  memset(&derived, 0, sizeof(bool));
  memset(&i, 0, sizeof(int));
  memset(ia, 0, sizeof(int));
  memset(&f, 0, sizeof(float));
  memset(fa, 0, sizeof(float));
  memset(&b, 0, sizeof(bool));
  memset(ba, 0, sizeof(bool));
  memset(&e, 0, sizeof(int));
  memset(&p.n, 0, sizeof(int));

  memcpy(&p, &p, sizeof(int));
  memcpy(&base, &base, sizeof(float));
  memcpy(&derived, &derived, sizeof(bool));
  memcpy(&i, &i, sizeof(int));
  memcpy(ia, ia, sizeof(int));
  memcpy(&f, &f, sizeof(float));
  memcpy(fa, fa, sizeof(float));
  memcpy(&b, &b, sizeof(bool));
  memcpy(ba, ba, sizeof(bool));
  memcpy(&e, &e, sizeof(int));
  memcpy(&p.n, &p.n, sizeof(int));

  memmove(&p, &p, sizeof(int));
  memmove(&base, &base, sizeof(float));
  memmove(&derived, &derived, sizeof(bool));
  memmove(&i, &i, sizeof(int));
  memmove(ia, ia, sizeof(int));
  memmove(&f, &f, sizeof(float));
  memmove(fa, fa, sizeof(float));
  memmove(&b, &b, sizeof(bool));
  memmove(ba, ba, sizeof(bool));
  memmove(&e, &e, sizeof(int));
  memmove(&p.n, &p.n, sizeof(int));
}
