// RUN: %check_clang_tidy %s -std=c++14 cert-mem57-cpp %t

namespace std {
typedef __typeof(sizeof(int)) size_t;
void *aligned_alloc(size_t, size_t);
void free(void *);
} // namespace std

struct alignas(128) Vector1 {
  char elems[128];
};

struct Vector2 {
  char elems[128];
};

struct alignas(128) Vector3 {
  char elems[128];
  static void *operator new(std::size_t nbytes) noexcept(true) {
    return std::aligned_alloc(alignof(Vector3), nbytes);
  }
  static void operator delete(void *p) {
    std::free(p);
  }
};

struct alignas(8) Vector4 {
  char elems[32];
};

void f() {
  auto *V1 = new Vector1;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: allocation function returns a pointer with alignment {{[0-9]+}} but the over-aligned type being allocated requires alignment 128 [cert-mem57-cpp]
  auto *V2 = new Vector2;
  auto *V3 = new Vector3;
  auto *V4 = new Vector4;
  auto *V1_Arr = new Vector1[2];
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: allocation function returns a pointer with alignment {{[0-9]+}} but the over-aligned type being allocated requires alignment 128 [cert-mem57-cpp]
}
