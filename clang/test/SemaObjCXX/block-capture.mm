// RUN: %clang_cc1 -std=c++23 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_23,cxx11_23,cxx23 %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_23,cxx11_23       %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_23,cxx11_23       %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -fobjc-arc -fblocks -Wno-c++11-extensions -verify=cxx98_23,cxx98          %s

#define TEST(T) void test_##T() { \
  __block T x;                    \
  (void)^(void) { (void)x; };     \
}

struct CopyOnly {
  CopyOnly();           // cxx23-note {{not viable}}
  CopyOnly(CopyOnly &); // cxx23-note {{not viable}}
};
TEST(CopyOnly); // cxx23-error {{no matching constructor}}

// Both ConstCopyOnly and NonConstCopyOnly are
// "pure" C++98 tests (pretend 'delete' means 'private').
// However we may extend implicit moves into C++98, we must make sure the
// results in these are not changed.
struct ConstCopyOnly {
  ConstCopyOnly();
  ConstCopyOnly(ConstCopyOnly &) = delete; // cxx98-note {{marked deleted here}}
  ConstCopyOnly(const ConstCopyOnly &);
};
TEST(ConstCopyOnly); // cxx98-error {{call to deleted constructor}}

struct NonConstCopyOnly {
  NonConstCopyOnly();
  NonConstCopyOnly(NonConstCopyOnly &);
  NonConstCopyOnly(const NonConstCopyOnly &) = delete; // cxx11_23-note {{marked deleted here}}
};
TEST(NonConstCopyOnly); // cxx11_23-error {{call to deleted constructor}}

struct CopyNoMove {
  CopyNoMove();
  CopyNoMove(CopyNoMove &);
  CopyNoMove(CopyNoMove &&) = delete; // cxx98_23-note {{marked deleted here}}
};
TEST(CopyNoMove); // cxx98_23-error {{call to deleted constructor}}

struct MoveOnly {
  MoveOnly();
  MoveOnly(MoveOnly &) = delete;
  MoveOnly(MoveOnly &&);
};
TEST(MoveOnly);

struct NoCopyNoMove {
  NoCopyNoMove();
  NoCopyNoMove(NoCopyNoMove &) = delete;
  NoCopyNoMove(NoCopyNoMove &&) = delete; // cxx98_23-note {{marked deleted here}}
};
TEST(NoCopyNoMove); // cxx98_23-error {{call to deleted constructor}}

struct ConvertingRVRef {
  ConvertingRVRef();
  ConvertingRVRef(ConvertingRVRef &) = delete;

  struct X {};
  ConvertingRVRef(X &&);
  operator X() const & = delete;
  operator X() &&;
};
TEST(ConvertingRVRef);

struct ConvertingCLVRef {
  ConvertingCLVRef();
  ConvertingCLVRef(ConvertingCLVRef &);

  struct X {};
  ConvertingCLVRef(X &&); // cxx98_23-note {{passing argument to parameter here}}
  operator X() const &;
  operator X() && = delete; // cxx98_23-note {{marked deleted here}}
};
TEST(ConvertingCLVRef); // cxx98_23-error {{invokes a deleted function}}

struct SubSubMove {};
struct SubMove : SubSubMove {
  SubMove();
  SubMove(SubMove &) = delete;

  SubMove(SubSubMove &&);
};
TEST(SubMove);


#if __cplusplus >= 202302L
// clang used to crash compiling this code.
namespace BlockInLambda {
  struct S {
    constexpr ~S();
  };

  void func(S const &a) {
    [a](auto b) {
      ^{
        (void)a;
      }();
    }(12);
  }
}
#endif
