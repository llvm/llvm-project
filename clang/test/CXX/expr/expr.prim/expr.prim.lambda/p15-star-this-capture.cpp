// RUN: %clang_cc1 -fsyntax-only -std=c++1z %s -verify

class NonCopyable {
  NonCopyable(const NonCopyable&) = delete; //expected-note3{{explicitly marked deleted here}}
  int x = 10;
  void foo() {
    auto L = [this] { return x; };
    const auto &M = [*this] { return x; };//expected-error{{call to deleted}}
    const auto &M2 = [this] () -> auto&& {
      ++x;
      return [*this] {  //expected-error{{call to deleted}} expected-warning{{reference to local}}
         return ++x; //expected-error{{read-only}}
      };
    };
    const auto &M3 = [*this] () mutable -> auto&& { //expected-error{{call to deleted}}
      ++x;
      return [this] {  // expected-warning{{reference to local}}
         return x;
      };
    };
  }
};
