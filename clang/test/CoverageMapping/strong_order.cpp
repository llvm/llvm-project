// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -std=c++23 -triple %itanium_abi_triple -main-file-name if.cpp %s

// No crash for following example.
// See https://github.com/llvm/llvm-project/issues/45481
namespace std {
class strong_ordering;

// Mock how STD defined unspecified parameters for the operators below.
struct _CmpUnspecifiedParam {
  consteval
  _CmpUnspecifiedParam(int _CmpUnspecifiedParam::*) noexcept {}
};

struct strong_ordering {
  signed char value;

  friend constexpr bool operator==(strong_ordering v,
                                   _CmpUnspecifiedParam) noexcept {
    return v.value == 0;
  }
  friend constexpr bool operator<(strong_ordering v,
                                  _CmpUnspecifiedParam) noexcept {
    return v.value < 0;
  }
  friend constexpr bool operator>(strong_ordering v,
                                  _CmpUnspecifiedParam) noexcept {
    return v.value > 0;
  }
  friend constexpr bool operator>=(strong_ordering v,
                                   _CmpUnspecifiedParam) noexcept {
    return v.value >= 0;
  }
  static const strong_ordering equal, greater, less;
};
constexpr strong_ordering strong_ordering::equal = {0};
constexpr strong_ordering strong_ordering::greater = {1};
constexpr strong_ordering strong_ordering::less = {-1};
} // namespace std

struct S {
    friend bool operator<(const S&, const S&);
    friend bool operator==(const S&, const S&);
};

struct MyStruct {
    friend bool operator==(MyStruct const& lhs, MyStruct const& rhs) = delete;
    friend std::strong_ordering operator<=>(MyStruct const& lhs, MyStruct const& rhs) = default;
    S value;
};

void foo(MyStruct bar){
    (void)(bar <=> bar);
}
