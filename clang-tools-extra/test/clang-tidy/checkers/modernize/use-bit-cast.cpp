// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-bit-cast %t

// CHECK-FIXES: #include <bit>

void *memcpy(void *To, const void *From, __SIZE_TYPE__ Size);

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

struct CommaSource {
  unsigned int Value;
};

struct CommaDestination {
  unsigned int Value;
};

void *memcpy(CommaDestination *, const CommaSource *, __SIZE_TYPE__);

enum class CommaSourceEnum : unsigned int {};
enum class CommaDestinationEnum : unsigned int {};

namespace rhs_adl {
struct Token {};
int operator,(unsigned int, Token);
} // namespace rhs_adl

namespace lhs_adl {
struct Token {};
int operator,(Token, unsigned int);
} // namespace lhs_adl

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

void sizeof_type_source_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(float));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void sizeof_type_destination_case() {
  float src = 1.0f;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(unsigned int));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void sizeof_dereferenced_source_pointer_case() {
  float src = 1.0f;
  float *srcp = &src;
  unsigned int dst;
  std::memcpy(&dst, &src, sizeof(*srcp));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
  std::memcpy(&dst, srcp, sizeof(*srcp));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

void sizeof_dereferenced_destination_pointer_case() {
  float src = 1.0f;
  unsigned int dst;
  unsigned int *dstp = &dst;
  std::memcpy(&dst, &src, sizeof(*dstp));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
  std::memcpy(dstp, &src, sizeof(*dstp));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
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

struct OneByte {
  unsigned char Value;
};

void anonymous_destination_case() {
  OneByte src{0};
  struct {
    unsigned char Value;
  } dst{};
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<decltype(dst)>(src);
}

struct AnonymousMemberHolder {
  struct {
    unsigned char Value;
  } dst;
};

void anonymous_member_destination_case() {
  OneByte src{0};
  AnonymousMemberHolder holder{};
  std::memcpy(&holder.dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: holder.dst = std::bit_cast<decltype(holder.dst)>(src);
}

void lambda_destination_case() {
  OneByte src{0};
  auto dst = [] {};
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<decltype(dst)>(src);
}

void lambda_reference_destination_case() {
  OneByte src{0};
  auto storage = [] {};
  auto &dst = storage;
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: std::memcpy(&dst, &src, sizeof(src));
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
  // CHECK-FIXES: int value = (dst = std::bit_cast<unsigned int>(src), 42);
}

void comma_rhs_case() {
  float src = 1.0f;
  unsigned int dst;
  (0, std::memcpy(&dst, &src, sizeof(src)));
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: (0, dst = std::bit_cast<unsigned int>(src));
}

void comma_record_destination_case() {
  CommaSource src{0};
  CommaDestination dst{0};
  int value = (std::memcpy(&dst, &src, sizeof(src)), 42);
  (void)value;
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: int value = ((void)(dst = std::bit_cast<CommaDestination>(src)), 42);
}

void comma_enum_destination_case() {
  CommaSourceEnum src{};
  CommaDestinationEnum dst{};
  int value = (std::memcpy(&dst, &src, sizeof(src)), 42);
  (void)value;
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: int value = ((void)(dst = std::bit_cast<CommaDestinationEnum>(src)), 42);
}

void comma_rhs_adl_case() {
  float src = 1.0f;
  unsigned int dst;
  auto value = (std::memcpy(&dst, &src, sizeof(src)), rhs_adl::Token{});
  (void)value;
  // CHECK-MESSAGES: :[[@LINE-2]]:17: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: auto value = ((void)(dst = std::bit_cast<unsigned int>(src)), rhs_adl::Token{});
}

void comma_lhs_adl_case() {
  float src = 1.0f;
  unsigned int dst;
  (lhs_adl::Token{}, std::memcpy(&dst, &src, sizeof(src)));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: (lhs_adl::Token{}, (void)(dst = std::bit_cast<unsigned int>(src)));
}

void nested_comma_adl_case() {
  float src = 1.0f;
  unsigned int dst;
  (0, (lhs_adl::Token{}, std::memcpy(&dst, &src, sizeof(src))));
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: (0, (lhs_adl::Token{}, (void)(dst = std::bit_cast<unsigned int>(src))));
}

void void_cast_case() {
  float src = 1.0f;
  unsigned int dst;
  (void)std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void void_cast_conditional_case(bool Cond) {
  float src = 1.0f;
  unsigned int dst;
  Cond ? (void)std::memcpy(&dst, &src, sizeof(src)) : (void)0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: Cond ? (void)(dst = std::bit_cast<unsigned int>(src)) : (void)0;
}

void void_cast_comma_case() {
  float src = 1.0f;
  unsigned int dst;
  ((void)std::memcpy(&dst, &src, sizeof(src)), rhs_adl::Token{});
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: ((void)(dst = std::bit_cast<unsigned int>(src)), rhs_adl::Token{});
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

void pointer_object_case(float *srcp) {
  unsigned int *dstp;
  std::memcpy(&dstp, &srcp, sizeof(srcp));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dstp = std::bit_cast<unsigned int *>(srcp);
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
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void volatile_destination_case() {
  float src = 1.0f;
  volatile unsigned int dst;
  std::memcpy(const_cast<unsigned int *>(&dst), &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

void volatile_record_destination_case(CommaSource src) {
  volatile CommaDestination dst{0};
  std::memcpy(const_cast<CommaDestination *>(&dst), &src, sizeof(src));
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

void global_overload_case() {
  CommaSource src{0};
  CommaDestination dst{0};
  memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
}

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

template <typename T>
void non_dependent_template_case() {
  float src = 1.0f;
  unsigned int dst;
  memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
  // CHECK-FIXES: dst = std::bit_cast<unsigned int>(src);
}

template void non_dependent_template_case<int>();

template <typename T>
concept MemcpyInRequires = requires(float &src, unsigned int &dst) {
  std::memcpy(&dst, &src, sizeof(src));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:3: warning: use 'std::bit_cast' instead of 'memcpy' for type punning
};

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
