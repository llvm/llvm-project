// RUN: %clang_cc1 -std=c++2b -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_2b,cxx11_2b,cxx2b %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_2b,cxx11_2b       %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_2b,cxx11_2b       %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -fobjc-arc -fblocks -Wno-c++11-extensions -verify=cxx98_2b,cxx98          %s

#define TEST(T) void test_##T() { \
  __block T x;                    \
  (void)^(void) { (void)x; };     \
}

struct CopyOnly {
  CopyOnly();           // cxx2b-note {{not viable}}
  CopyOnly(CopyOnly &); // cxx2b-note {{not viable}}
};
TEST(CopyOnly); // cxx2b-error {{no matching constructor}}

struct ConstCopyOnly {
  ConstCopyOnly();
  ConstCopyOnly(ConstCopyOnly &) = delete; // cxx98-note {{marked deleted here}}
  ConstCopyOnly(const ConstCopyOnly &);
};
TEST(ConstCopyOnly); // cxx98-error {{call to deleted constructor}}

struct NonConstCopyOnly {
  NonConstCopyOnly();
  NonConstCopyOnly(NonConstCopyOnly &);
  NonConstCopyOnly(const NonConstCopyOnly &) = delete; // cxx11_2b-note {{marked deleted here}}
};
TEST(NonConstCopyOnly); // cxx11_2b-error {{call to deleted constructor}}

struct CopyNoMove {
  CopyNoMove();
  CopyNoMove(CopyNoMove &);
  CopyNoMove(CopyNoMove &&) = delete; // cxx11_2b-note {{marked deleted here}}
};
TEST(CopyNoMove); // cxx11_2b-error {{call to deleted constructor}}

struct MoveOnly {
  MoveOnly();
  MoveOnly(MoveOnly &) = delete; // cxx98-note {{marked deleted here}}
  MoveOnly(MoveOnly &&);
};
TEST(MoveOnly); // cxx98-error {{call to deleted constructor}}

struct NoCopyNoMove {
  NoCopyNoMove();
  NoCopyNoMove(NoCopyNoMove &) = delete;  // cxx98-note {{marked deleted here}}
  NoCopyNoMove(NoCopyNoMove &&) = delete; // cxx11_2b-note {{marked deleted here}}
};
TEST(NoCopyNoMove); // cxx98_2b-error {{call to deleted constructor}}

struct ConvertingRVRef {
  ConvertingRVRef();
  ConvertingRVRef(ConvertingRVRef &) = delete; // cxx98-note {{marked deleted here}}

  struct X {};
  ConvertingRVRef(X &&);
  operator X() const & = delete;
  operator X() &&;
};
TEST(ConvertingRVRef); // cxx98-error {{call to deleted constructor}}

struct ConvertingCLVRef {
  ConvertingCLVRef();
  ConvertingCLVRef(ConvertingCLVRef &);

  struct X {};
  ConvertingCLVRef(X &&); // cxx11_2b-note {{passing argument to parameter here}}
  operator X() const &;
  operator X() && = delete; // cxx11_2b-note {{marked deleted here}}
};
TEST(ConvertingCLVRef); // cxx11_2b-error {{invokes a deleted function}}

struct SubSubMove {};
struct SubMove : SubSubMove {
  SubMove();
  SubMove(SubMove &) = delete; // cxx98-note {{marked deleted here}}

  SubMove(SubSubMove &&);
};
TEST(SubMove); // cxx98-error {{call to deleted constructor}}
