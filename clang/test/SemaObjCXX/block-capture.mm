// RUN: %clang_cc1 -std=c++2b -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_2b,cxx20_2b,cxx2b %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_2b,cxx20_2b       %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_2b,cxx98_11       %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -fobjc-arc -fblocks -Wno-c++11-extensions -verify=cxx98_2b,cxx98_11       %s

#define TEST(T) void test_##T() { \
  __block T x;                    \
  (void)^(void) { (void)x; };     \
}

struct CopyOnly {
  CopyOnly();           // cxx2b-note {{not viable}}
  CopyOnly(CopyOnly &); // cxx2b-note {{not viable}}
};
TEST(CopyOnly); // cxx2b-error {{no matching constructor}}

struct CopyNoMove {
  CopyNoMove();
  CopyNoMove(CopyNoMove &);
  CopyNoMove(CopyNoMove &&) = delete; // cxx98_2b-note {{marked deleted here}}
};
TEST(CopyNoMove); // cxx98_2b-error {{call to deleted constructor}}

struct MoveOnly {
  MoveOnly();
  MoveOnly(MoveOnly &) = delete;
  MoveOnly(MoveOnly &&);
};
TEST(MoveOnly);

struct NoCopyNoMove {
  NoCopyNoMove();
  NoCopyNoMove(NoCopyNoMove &) = delete;
  NoCopyNoMove(NoCopyNoMove &&) = delete; // cxx98_2b-note {{marked deleted here}}
};
TEST(NoCopyNoMove); // cxx98_2b-error {{call to deleted constructor}}

struct ConvertingRVRef {
  ConvertingRVRef();
  ConvertingRVRef(ConvertingRVRef &) = delete; // cxx98_11-note {{marked deleted here}}

  struct X {};
  ConvertingRVRef(X &&);
  operator X() const & = delete;
  operator X() &&;
};
TEST(ConvertingRVRef); // cxx98_11-error {{call to deleted constructor}}

struct ConvertingCLVRef {
  ConvertingCLVRef();
  ConvertingCLVRef(ConvertingCLVRef &);

  struct X {};
  ConvertingCLVRef(X &&); // cxx20_2b-note {{passing argument to parameter here}}
  operator X() const &;
  operator X() && = delete; // cxx20_2b-note {{marked deleted here}}
};
TEST(ConvertingCLVRef); // cxx20_2b-error {{invokes a deleted function}}

struct SubSubMove {};
struct SubMove : SubSubMove {
  SubMove();
  SubMove(SubMove &) = delete; // cxx98_11-note {{marked deleted here}}

  SubMove(SubSubMove &&);
};
TEST(SubMove); // cxx98_11-error {{call to deleted constructor}}
