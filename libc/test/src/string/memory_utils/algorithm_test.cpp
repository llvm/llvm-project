#define LLVM_LIBC_USE_BUILTIN_MEMCPY_INLINE 0
#define LLVM_LIBC_USE_BUILTIN_MEMSET_INLINE 0

#include "utils/UnitTest/Test.h"
#include <src/__support/CPP/Array.h>
#include <src/string/memory_utils/algorithm.h>
#include <src/string/memory_utils/backends.h>

#include <sstream>

namespace __llvm_libc {

struct alignas(64) Buffer : cpp::Array<char, 128> {
  bool contains(const char *ptr) const {
    return ptr >= data() && ptr < (data() + size());
  }
  size_t getOffset(const char *ptr) const { return ptr - data(); }
  void fill(char c) {
    for (auto itr = begin(); itr != end(); ++itr)
      *itr = c;
  }
};

static Buffer buffer1;
static Buffer buffer2;
static std::ostringstream LOG;

struct TestBackend {
  static constexpr bool IS_BACKEND_TYPE = true;

  template <typename T> static void log(const char *Action, const char *ptr) {
    LOG << Action << "<" << sizeof(T) << "> ";
    if (buffer1.contains(ptr))
      LOG << "a[" << buffer1.getOffset(ptr) << "]";
    else if (buffer2.contains(ptr))
      LOG << "b[" << buffer2.getOffset(ptr) << "]";
    LOG << "\n";
  }

  template <typename T, Temporality TS, Aligned AS>
  static T load(const T *src) {
    log<T>((AS == Aligned::YES ? "LdA" : "LdU"),
           reinterpret_cast<const char *>(src));
    return Scalar64BitBackend::load<T, TS, AS>(src);
  }

  template <typename T, Temporality TS, Aligned AS>
  static void store(T *dst, T value) {
    log<T>((AS == Aligned::YES ? "StA" : "StU"),
           reinterpret_cast<const char *>(dst));
    Scalar64BitBackend::store<T, TS, AS>(dst, value);
  }

  template <typename T> static inline T splat(ubyte value) {
    LOG << "Splat<" << sizeof(T) << "> " << (unsigned)value << '\n';
    return Scalar64BitBackend::splat<T>(value);
  }

  template <typename T> static inline uint64_t notEquals(T v1, T v2) {
    LOG << "Neq<" << sizeof(T) << ">\n";
    return Scalar64BitBackend::notEquals<T>(v1, v2);
  }

  template <typename T> static inline int32_t threeWayCmp(T v1, T v2) {
    LOG << "Diff<" << sizeof(T) << ">\n";
    return Scalar64BitBackend::threeWayCmp<T>(v1, v2);
  }

  template <size_t Size>
  using getNextType = Scalar64BitBackend::getNextType<Size>;
};

struct LlvmLibcAlgorithm : public testing::Test {
  void SetUp() override {
    LOG = std::ostringstream();
    LOG << '\n';
  }

  void fillEqual() {
    buffer1.fill('a');
    buffer2.fill('a');
  }

  void fillDifferent() {
    buffer1.fill('a');
    buffer2.fill('b');
  }

  const char *getTrace() {
    trace_ = LOG.str();
    return trace_.c_str();
  }

  const char *stripComments(const char *expected) {
    expected_.clear();
    std::stringstream ss(expected);
    std::string line;
    while (std::getline(ss, line, '\n')) {
      const auto pos = line.find('#');
      if (pos == std::string::npos) {
        expected_ += line;
      } else {
        auto log = line.substr(0, pos);
        while (!log.empty() && std::isspace(log.back()))
          log.pop_back();
        expected_ += log;
      }
      expected_ += '\n';
    }
    return expected_.c_str();
  }

  template <size_t Align = 1> SrcAddr<Align> buf1(size_t offset = 0) const {
    return buffer1.data() + offset;
  }
  template <size_t Align = 1> SrcAddr<Align> buf2(size_t offset = 0) const {
    return buffer2.data() + offset;
  }
  template <size_t Align = 1> DstAddr<Align> dst(size_t offset = 0) const {
    return buffer1.data() + offset;
  }
  template <size_t Align = 1> SrcAddr<Align> src(size_t offset = 0) const {
    return buffer2.data() + offset;
  }

private:
  std::string trace_;
  std::string expected_;
};

using _8 = SizedOp<TestBackend, 8>;

///////////////////////////////////////////////////////////////////////////////
//// Testing fixed fized forward operations
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Copy

TEST_F(LlvmLibcAlgorithm, copy_1) {
  SizedOp<TestBackend, 1>::copy(dst(), src());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<1> b[0]
StU<1> a[0]
)"));
}

TEST_F(LlvmLibcAlgorithm, copy_15) {
  SizedOp<TestBackend, 15>::copy(dst(), src());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StU<8> a[0]
LdU<4> b[8]
StU<4> a[8]
LdU<2> b[12]
StU<2> a[12]
LdU<1> b[14]
StU<1> a[14]
)"));
}

TEST_F(LlvmLibcAlgorithm, copy_16) {
  SizedOp<TestBackend, 16>::copy(dst(), src());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StU<8> a[0]
LdU<8> b[8]
StU<8> a[8]
)"));
}

///////////////////////////////////////////////////////////////////////////////
// Move

TEST_F(LlvmLibcAlgorithm, move_1) {
  SizedOp<TestBackend, 1>::move(dst(), src());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<1> b[0]
StU<1> a[0]
)"));
}

TEST_F(LlvmLibcAlgorithm, move_15) {
  SizedOp<TestBackend, 15>::move(dst(), src());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
LdU<4> b[8]
LdU<2> b[12]
LdU<1> b[14]
StU<1> a[14]
StU<2> a[12]
StU<4> a[8]
StU<8> a[0]
)"));
}

TEST_F(LlvmLibcAlgorithm, move_16) {
  SizedOp<TestBackend, 16>::move(dst(), src());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
LdU<8> b[8]
StU<8> a[8]
StU<8> a[0]
)"));
}

///////////////////////////////////////////////////////////////////////////////
// set

TEST_F(LlvmLibcAlgorithm, set_1) {
  SizedOp<TestBackend, 1>::set(dst(), ubyte{42});
  EXPECT_STREQ(getTrace(), stripComments(R"(
Splat<1> 42
StU<1> a[0]
)"));
}

TEST_F(LlvmLibcAlgorithm, set_15) {
  SizedOp<TestBackend, 15>::set(dst(), ubyte{42});
  EXPECT_STREQ(getTrace(), stripComments(R"(
Splat<8> 42
StU<8> a[0]
Splat<4> 42
StU<4> a[8]
Splat<2> 42
StU<2> a[12]
Splat<1> 42
StU<1> a[14]
)"));
}

TEST_F(LlvmLibcAlgorithm, set_16) {
  SizedOp<TestBackend, 16>::set(dst(), ubyte{42});
  EXPECT_STREQ(getTrace(), stripComments(R"(
Splat<8> 42
StU<8> a[0]
Splat<8> 42
StU<8> a[8]
)"));
}

///////////////////////////////////////////////////////////////////////////////
// different

TEST_F(LlvmLibcAlgorithm, different_1) {
  fillEqual();
  SizedOp<TestBackend, 1>::isDifferent(buf1(), buf2());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<1> a[0]
LdU<1> b[0]
Neq<1>
)"));
}

TEST_F(LlvmLibcAlgorithm, different_15) {
  fillEqual();
  SizedOp<TestBackend, 15>::isDifferent(buf1(), buf2());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[0]
LdU<8> b[0]
Neq<8>
LdU<4> a[8]
LdU<4> b[8]
Neq<4>
LdU<2> a[12]
LdU<2> b[12]
Neq<2>
LdU<1> a[14]
LdU<1> b[14]
Neq<1>
)"));
}

TEST_F(LlvmLibcAlgorithm, different_15_no_shortcircuit) {
  fillDifferent();
  SizedOp<TestBackend, 15>::isDifferent(buf1(), buf2());
  // If buffer compare isDifferent we continue to aggregate.
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[0]
LdU<8> b[0]
Neq<8>
LdU<4> a[8]
LdU<4> b[8]
Neq<4>
LdU<2> a[12]
LdU<2> b[12]
Neq<2>
LdU<1> a[14]
LdU<1> b[14]
Neq<1>
)"));
}

TEST_F(LlvmLibcAlgorithm, different_16) {
  fillEqual();
  SizedOp<TestBackend, 16>::isDifferent(buf1(), buf2());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[0]
LdU<8> b[0]
Neq<8>
LdU<8> a[8]
LdU<8> b[8]
Neq<8>
)"));
}

///////////////////////////////////////////////////////////////////////////////
// three_way_cmp

TEST_F(LlvmLibcAlgorithm, three_way_cmp_eq_1) {
  fillEqual();
  SizedOp<TestBackend, 1>::threeWayCmp(buf1(), buf2());
  // Buffer compare equal, returning 0 and no call to Diff.
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<1> a[0]
LdU<1> b[0]
Diff<1>
)"));
}

TEST_F(LlvmLibcAlgorithm, three_way_cmp_eq_15) {
  fillEqual();
  SizedOp<TestBackend, 15>::threeWayCmp(buf1(), buf2());
  // Buffer compare equal, returning 0 and no call to Diff.
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[0]
LdU<8> b[0]
Diff<8>
LdU<4> a[8]
LdU<4> b[8]
Diff<4>
LdU<2> a[12]
LdU<2> b[12]
Diff<2>
LdU<1> a[14]
LdU<1> b[14]
Diff<1>
)"));
}

TEST_F(LlvmLibcAlgorithm, three_way_cmp_neq_15_shortcircuit) {
  fillDifferent();
  SizedOp<TestBackend, 16>::threeWayCmp(buf1(), buf2());
  // If buffer compare isDifferent we stop early.
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[0]
LdU<8> b[0]
Diff<8>
)"));
}

TEST_F(LlvmLibcAlgorithm, three_way_cmp_eq_16) {
  fillEqual();
  SizedOp<TestBackend, 16>::threeWayCmp(buf1(), buf2());
  // Buffer compare equal, returning 0 and no call to Diff.
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[0]
LdU<8> b[0]
Diff<8>
LdU<8> a[8]
LdU<8> b[8]
Diff<8>
)"));
}

///////////////////////////////////////////////////////////////////////////////
//// Testing skip operations
///////////////////////////////////////////////////////////////////////////////

TEST_F(LlvmLibcAlgorithm, skip_and_set) {
  Skip<11>::Then<SizedOp<TestBackend, 1>>::set(dst(), ubyte{42});
  EXPECT_STREQ(getTrace(), stripComments(R"(
Splat<1> 42
StU<1> a[11]
)"));
}

TEST_F(LlvmLibcAlgorithm, skip_and_different_1) {
  Skip<11>::Then<SizedOp<TestBackend, 1>>::isDifferent(buf1(), buf2());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<1> a[11]
LdU<1> b[11]
Neq<1>
)"));
}

TEST_F(LlvmLibcAlgorithm, skip_and_three_way_cmp_8) {
  Skip<11>::Then<SizedOp<TestBackend, 1>>::threeWayCmp(buf1(), buf2());
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<1> a[11]
LdU<1> b[11]
Diff<1>
)"));
}

///////////////////////////////////////////////////////////////////////////////
//// Testing tail operations
///////////////////////////////////////////////////////////////////////////////

TEST_F(LlvmLibcAlgorithm, tail_copy_8) {
  Tail<_8>::copy(dst(), src(), 16);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[8]
StU<8> a[8]
)"));
}

TEST_F(LlvmLibcAlgorithm, tail_move_8) {
  Tail<_8>::move(dst(), src(), 16);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[8]
StU<8> a[8]
)"));
}

TEST_F(LlvmLibcAlgorithm, tail_set_8) {
  Tail<_8>::set(dst(), ubyte{42}, 16);
  EXPECT_STREQ(getTrace(), stripComments(R"(
Splat<8> 42
StU<8> a[8]
)"));
}

TEST_F(LlvmLibcAlgorithm, tail_different_8) {
  fillEqual();
  Tail<_8>::isDifferent(buf1(), buf2(), 16);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[8]
LdU<8> b[8]
Neq<8>
)"));
}

TEST_F(LlvmLibcAlgorithm, tail_three_way_cmp_8) {
  fillEqual();
  Tail<_8>::threeWayCmp(buf1(), buf2(), 16);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> a[8]
LdU<8> b[8]
Diff<8>
)"));
}

///////////////////////////////////////////////////////////////////////////////
//// Testing HeadTail operations
///////////////////////////////////////////////////////////////////////////////

TEST_F(LlvmLibcAlgorithm, head_tail_copy_8) {
  HeadTail<_8>::copy(dst(), src(), 16);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StU<8> a[0]
LdU<8> b[8]
StU<8> a[8]
)"));
}

///////////////////////////////////////////////////////////////////////////////
//// Testing Loop operations
///////////////////////////////////////////////////////////////////////////////

TEST_F(LlvmLibcAlgorithm, loop_copy_one_iteration_and_tail) {
  Loop<_8>::copy(dst(), src(), 10);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StU<8> a[0] # covers 0-7
LdU<8> b[2]
StU<8> a[2] # covers 2-9
)"));
}

TEST_F(LlvmLibcAlgorithm, loop_copy_two_iteration_and_tail) {
  Loop<_8>::copy(dst(), src(), 17);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StU<8> a[0] # covers 0-7
LdU<8> b[8]
StU<8> a[8] # covers 8-15
LdU<8> b[9]
StU<8> a[9] # covers 9-16
)"));
}

TEST_F(LlvmLibcAlgorithm, loop_with_one_turn_is_inefficient_but_ok) {
  Loop<_8>::copy(dst(), src(), 8);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StU<8> a[0] # first iteration covers 0-7
LdU<8> b[0] # tail also covers 0-7 but since Loop is supposed to be used
StU<8> a[0] # with a sufficient number of iterations the tail cost is amortised
)"));
}

TEST_F(LlvmLibcAlgorithm, loop_with_round_number_of_turn) {
  Loop<_8>::copy(dst(), src(), 24);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StU<8> a[0] # first iteration covers 0-7
LdU<8> b[8]
StU<8> a[8] # second iteration covers 8-15
LdU<8> b[16]
StU<8> a[16]
)"));
}

TEST_F(LlvmLibcAlgorithm, dst_aligned_loop) {
  Loop<_8>::copy(dst<16>(), src(), 23);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[0]
StA<8> a[0] # store is aligned on 16B
LdU<8> b[8]
StA<8> a[8] # subsequent stores are aligned
LdU<8> b[15]
StU<8> a[15] # Tail is always unaligned
)"));
}

TEST_F(LlvmLibcAlgorithm, aligned_loop) {
  Loop<_8>::copy(dst<16>(), src<8>(), 23);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdA<8> b[0] # load is aligned on 8B
StA<8> a[0] # store is aligned on 16B
LdA<8> b[8] # subsequent loads are aligned
StA<8> a[8] # subsequent stores are aligned
LdU<8> b[15] # Tail is always unaligned
StU<8> a[15] # Tail is always unaligned
)"));
}

///////////////////////////////////////////////////////////////////////////////
//// Testing Align operations
///////////////////////////////////////////////////////////////////////////////

TEST_F(LlvmLibcAlgorithm, align_dst_copy_8) {
  Align<_8, Arg::Dst>::Then<Loop<_8>>::copy(dst(2), src(3), 31);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[3]
StU<8> a[2] # First store covers unaligned bytes
LdU<8> b[9]
StA<8> a[8] # First aligned store
LdU<8> b[17]
StA<8> a[16] # Subsequent stores are aligned
LdU<8> b[25]
StA<8> a[24] # Subsequent stores are aligned
LdU<8> b[26]
StU<8> a[25] # Last store covers remaining bytes
)"));
}

TEST_F(LlvmLibcAlgorithm, align_src_copy_8) {
  Align<_8, Arg::Src>::Then<Loop<_8>>::copy(dst(2), src(3), 31);
  EXPECT_STREQ(getTrace(), stripComments(R"(
LdU<8> b[3] # First load covers unaligned bytes
StU<8> a[2]
LdA<8> b[8] # First aligned load
StU<8> a[7]
LdA<8> b[16] # Subsequent loads are aligned
StU<8> a[15]
LdA<8> b[24] # Subsequent loads are aligned
StU<8> a[23]
LdU<8> b[26] # Last load covers remaining bytes
StU<8> a[25]
)"));
}

} // namespace __llvm_libc
