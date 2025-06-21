// RUN: %clangxx -std=c++23 -fsyntax-only -Xclang -verify %s

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <optional>
#include <variant>
#include <array>
#include <span>

static std::vector<std::string> getVector() {
  return {"first", "second", "third"};
}

static std::map<std::string, std::vector<int>> getMap() {
  return {{"key", {1, 2, 3}}};
}

static std::tuple<std::vector<double>> getTuple() {
  return std::make_tuple(std::vector<double>{3.14, 2.71});
}

static std::optional<std::vector<char>> getOptionalColl() {
  return std::vector<char>{'x', 'y', 'z'};
}

static std::variant<std::string, int> getVariant() {
  return std::string("variant");
}

static const std::array<int, 4>& arrOfConst() {
  static const std::array<int, 4> arr = {10, 20, 30, 40};
  return arr;
}

static void testGetVectorSubscript() {
  for (auto e : getVector()[0]) {
    (void)e;
  }
}

static void testGetMapSubscript() {
  for (auto valueElem : getMap()["key"]) {
    (void)valueElem;
  }
}

static void testGetTuple() {
  for (auto e : std::get<0>(getTuple())) {
    (void)e;
  }
}

static void testOptionalValue() {
  for (auto e : getOptionalColl().value()) {
    (void)e;
  }
}

static void testVariantGetString() {
  for (char c : std::get<std::string>(getVariant())) {
    (void)c;
  }
}

static void testSpanLastFromConstArray() {
  for (auto s : std::span{arrOfConst()}.last(2)) {
    (void)s;
  }
}

static void testSpanFromVectorPtr() {
  for (auto e : std::span(getVector().data(), 2)) {
    (void)e;
  }
}

// expected-no-diagnostics
