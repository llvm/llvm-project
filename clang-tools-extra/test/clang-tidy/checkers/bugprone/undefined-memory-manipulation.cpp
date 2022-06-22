// RUN: %check_clang_tidy %s bugprone-undefined-memory-manipulation %t

void *memset(void *, int, __SIZE_TYPE__);
void *memcpy(void *, const void *, __SIZE_TYPE__);
void *memmove(void *, const void *, __SIZE_TYPE__);

namespace std {
using ::memcpy;
using ::memmove;
using ::memset;
}

namespace types {
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

} // end namespace types

void f(types::NoDef *s) {
  memset(s, 0, 5);
}

template <typename T>
void memset_temp(T *b) {
  memset(b, 0, sizeof(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::VirtualFunc' is not TriviallyCopyable [bugprone-undefined-memory-manipulation]
}

template <typename S, typename T>
void memcpy_temp(S *a, T *b) {
  memcpy(a, b, sizeof(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::VirtualFunc'
}

template <typename S, typename T>
void memmove_temp(S *a, T *b) {
  memmove(a, b, sizeof(T));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::VirtualFunc'
}

namespace aliases {
using Copy2 = types::Copy;
typedef types::Move Move2;
}

void notTriviallyCopyable() {
  types::Plain p; // TriviallyCopyable for variety
  types::Destruct d;
  types::Copy c;
  types::Move m;
  types::VirtualFunc vf;
  types::VirtualBase vb;

  memset(&vf, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::VirtualFunc'
  memset(&d, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::Destruct'
  memset(&c, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::Copy'
  std::memset(&m, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::Move'
  ::memset(&vb, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::VirtualBase'

  memcpy(&p, &vf, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::VirtualFunc'
  memcpy(&p, &d, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::Destruct'
  memcpy(&c, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::Copy'
  std::memcpy(&m, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::Move'
  ::memcpy(&vb, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::VirtualBase'

  memmove(&vf, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::VirtualFunc'
  memmove(&d, &p, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::Destruct'
  memmove(&p, &c, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::Copy'
  std::memmove(&p, &m, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::Move'
  ::memmove(&p, &vb, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::VirtualBase'

#define MEMSET memset(&vf, 0, sizeof(int));
  MEMSET
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::VirtualFunc'
#define MEMCPY memcpy(&d, &p, sizeof(int));
  MEMCPY
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'types::Destruct'
#define MEMMOVE memmove(&p, &c, sizeof(int));
  MEMMOVE
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, source object type 'types::Copy'

  memset_temp<types::VirtualFunc>(&vf);
  memcpy_temp<types::Plain, types::VirtualFunc>(&p, &vf);
  memmove_temp<types::Plain, types::VirtualFunc>(&p, &vf);

  aliases::Copy2 c2;
  aliases::Move2 m2;
  memset(&c2, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'aliases::Copy2'
  memset(&m2, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'aliases::Move2'

  typedef aliases::Copy2 Copy3;
  typedef aliases::Copy2 *PCopy2;
  typedef Copy3 *PCopy3;
  Copy3 c3;
  PCopy2 pc2;
  PCopy3 pc3;
  memset(&c3, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'Copy3'
  memset(pc2, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'aliases::Copy2'
  memset(pc3, 0, sizeof(int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: undefined behavior, destination object type 'Copy3'
}

void triviallyCopyable() {
  types::Plain p;
  types::Base base;
  types::Derived derived;

  int i = 5;
  int ia[3] = {1, 2, 3};
  float f = 3.14;
  float fa[3] = {1.1, 2.2, 3.3};
  bool b = false;
  bool ba[2] = {true, false};
  types::E e = types::X;
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
