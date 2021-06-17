// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_20,cxx20    %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fobjc-arc -fblocks                       -verify=cxx98_20,cxx98_11 %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -fobjc-arc -fblocks -Wno-c++11-extensions -verify=cxx98_20,cxx98_11 %s

#define TEST(T) void test_##T() { \
  __block T x;                    \
  (void)^(void) { (void)x; };     \
}

struct CopyOnly {
  CopyOnly();
  CopyOnly(CopyOnly &);
};
TEST(CopyOnly);

struct CopyNoMove {
  CopyNoMove();
  CopyNoMove(CopyNoMove &);
  CopyNoMove(CopyNoMove &&) = delete; // cxx98_20-note {{marked deleted here}}
};
TEST(CopyNoMove); // cxx98_20-error {{call to deleted constructor}}

struct MoveOnly {
  MoveOnly();
  MoveOnly(MoveOnly &) = delete;
  MoveOnly(MoveOnly &&);
};
TEST(MoveOnly);

struct NoCopyNoMove {
  NoCopyNoMove();
  NoCopyNoMove(NoCopyNoMove &) = delete;
  NoCopyNoMove(NoCopyNoMove &&) = delete; // cxx98_20-note {{marked deleted here}}
};
TEST(NoCopyNoMove); // cxx98_20-error {{call to deleted constructor}}

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
  ConvertingCLVRef(X &&); // cxx20-note {{passing argument to parameter here}}
  operator X() const &;
  operator X() && = delete; // cxx20-note {{marked deleted here}}
};
TEST(ConvertingCLVRef); // cxx20-error {{invokes a deleted function}}

struct SubSubMove {};
struct SubMove : SubSubMove {
  SubMove();
  SubMove(SubMove &) = delete; // cxx98_11-note {{marked deleted here}}

  SubMove(SubSubMove &&);
};
TEST(SubMove); // cxx98_11-error {{call to deleted constructor}}
