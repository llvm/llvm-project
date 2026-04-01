// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-bit-cast %t

// CHECK-FIXES: #include <bit>

void *memcpy(void *To, const void *From, unsigned long long Size);

namespace std {
template <typename T, unsigned long long N>
struct array {
  T Storage[N];
};

using ::memcpy;
}

template <typename T>
struct identity {
  using type = T;
};

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial &);
  int Value;
};

extern unsigned long long n;

void basic_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning [modernize-use-bit-cast]
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void unqualified_case() {
  float src = 1.0f;
  unsigned int dst;
  memcpy(&dst, &src, sizeof(dst));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void global_case() {
  float src = 1.0f;
  unsigned int dst;
  ::memcpy(&dst, &src, sizeof(unsigned int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void explicit_cast_case() {
  float src = 1.0f;
  unsigned int dst = 0;
  std::memcpy(static_cast<void *>(&dst), static_cast<const void *>(&src),
              sizeof(dst));
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void alias_case() {
  using U = identity<unsigned int>::type;
  using F = identity<float>::type;
  F src = 1.0f;
  U dst;
  std::memcpy(&dst, &src, sizeof(U));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<U>(src);
}

void const_source_case() {
  const float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void std_array_case() {
  std::array<float, 1> src{{1.0f}};
  std::array<unsigned int, 1> dst{};
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<std::array<unsigned int, 1>>(src);
}

void raw_array_source_case() {
  float src[1] = {1.0f};
  std::array<unsigned int, 1> dst{};
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<std::array<unsigned int, 1>>(src);
}

void lambda_case() {
  auto L = [] {
    float src = 1.0f;
    unsigned int dst;
    std::memcpy(&dst, &src, sizeof(src));
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
    // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
  };
  L();
}

void if_body_case(bool Cond) {
  float src = 1.0f;
  unsigned int dst;
  if (Cond)
    std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: if (Cond)
  // CHECK-FIXES-NEXT: dst = std::bit_cast<unsigned int>(src);
}

void comma_lhs_case() {
  float src = 1.0f;
  unsigned int dst;
  int value = (std::memcpy(&dst, &src, sizeof(src)), 42);
  (void)value;
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: int value = ((void)(dst = std::bit_cast<unsigned int>(src)), 42);
}

void void_cast_case() {
  float src = 1.0f;
  unsigned int dst;
  (void)std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: (void)(dst = std::bit_cast<unsigned int>(src));
}

void same_type_case() {
  float src = 1.0f;
  float dst = 0.0f;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void pointer_case(int *srcp) {
  int *dstp;
  std::memcpy(&dstp, &srcp, sizeof(srcp));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void array_case() {
  unsigned char bytes[sizeof(float)];
  float src = 1.0f;
  std::memcpy(bytes, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void raw_array_destination_case() {
  std::array<float, 1> src{{1.0f}};
  unsigned int dst[1];
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void buffer_pointer_case(float *srcp, unsigned int *dstp) {
  std::memcpy(dstp, srcp, sizeof(*srcp));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void partial_copy_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, 2);
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void unknown_copy_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, n);
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void non_trivial_case(NonTrivial src) {
  NonTrivial dst;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void volatile_case() {
  volatile float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, const_cast<const float *>(&src), sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

struct Wrap {
  float src;
  unsigned int dst;
};

struct SourceStruct {
  int Value;
};

struct DestStruct {
  const int Value;
};

void member_case() {
  Wrap W{1.0f, 0};
  std::memcpy(&W.dst, &W.src, sizeof(W.src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: W.dst = std::bit_cast<unsigned int>(W.src);
}

void pointer_member_case(Wrap *P) {
  std::memcpy(&P->dst, &P->src, sizeof(P->src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: P->dst = std::bit_cast<unsigned int>(P->src);
}

void member_pointer_case(Wrap W, float Wrap::*Src, unsigned int Wrap::*Dst) {
  std::memcpy(&(W.*Dst), &(W.*Src), sizeof(W.*Src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: W.*Dst = std::bit_cast<unsigned int>(W.*Src);
}

void pointer_member_pointer_case(Wrap *P, float Wrap::*Src,
                                 unsigned int Wrap::*Dst) {
  std::memcpy(&(P->*Dst), &(P->*Src), sizeof(P->*Src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: P->*Dst = std::bit_cast<unsigned int>(P->*Src);
}

void builtin_case() {
  float src = 1.0f;
  unsigned int dst;
  __builtin_memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

namespace ns {
struct A {
  unsigned int Value;
};

struct B {
  unsigned int Value;
};

void memcpy(B *, const A *, unsigned long long);

void overload_case() {
  A src{0};
  B dst{0};
  memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}
} // namespace ns

#define DO_COPY(Dst, Src) std::memcpy(&(Dst), &(Src), sizeof(Src))

void macro_case() {
  float src = 1.0f;
  unsigned int dst;
  DO_COPY(dst, src);
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

template <typename To, typename From>
requires(sizeof(To) == sizeof(From))
To template_case(From src) {
  To dst;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  return dst;
}

void unevaluated_case() {
  float src = 1.0f;
  unsigned int dst;
  (void)sizeof(std::memcpy(&dst, &src, sizeof(src)));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:16: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void used_return_value_case() {
  float src = 1.0f;
  unsigned int dst;
  void *Ptr = std::memcpy(&dst, &src, sizeof(src));
  (void)Ptr;
  // CHECK-MESSAGES-NOT: :[[@LINE-2]]:15: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void comma_rhs_used_case() {
  float src = 1.0f;
  unsigned int dst;
  void *Ptr = (0, std::memcpy(&dst, &src, sizeof(src)));
  (void)Ptr;
  // CHECK-MESSAGES-NOT: :[[@LINE-2]]:19: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void deleted_assignment_case(SourceStruct src) {
  DestStruct dst{0};
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void condition_use_case() {
  float src = 1.0f;
  unsigned int dst;
  if (std::memcpy(&dst, &src, sizeof(src)))
    (void)0;
  // CHECK-MESSAGES-NOT: :[[@LINE-2]]:7: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void conditional_operand_case(bool Cond) {
  float src = 1.0f;
  unsigned int dst;
  void *Ptr = nullptr;
  (void)(Cond ? std::memcpy(&dst, &src, sizeof(src)) : Ptr);
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:17: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}
