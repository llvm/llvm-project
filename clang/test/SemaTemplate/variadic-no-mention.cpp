// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template <class... T>
struct Types {};
template <int& field>
using Forget = int;
template <int&... fields>
using SeqKey = Types<Forget<fields>...>;

template <typename Key, typename Value>
struct HelperBase {
  using ResponseParser = Key();
  HelperBase(ResponseParser response_parser) {}
};
template <int&... fields>
SeqKey<fields...> Parser();

template <int&... fields>
struct Helper : public HelperBase<SeqKey<fields...>, double> {
  using Key = SeqKey<fields...>;
  using Value = double;
  using ParentClass = HelperBase<Key, Value>;
  Helper() : ParentClass(Parser<fields...>) {}
};

void test() {
  Helper<>();
}

