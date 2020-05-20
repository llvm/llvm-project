//===--- MarshallingTests.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../TestTU.h"
#include "index/Serialization.h"
#include "index/remote/marshalling/Marshalling.h"
#include "llvm/Support/StringSaver.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {
namespace remote {
namespace {

TEST(RemoteMarshallingTest, SymbolSerialization) {
  const auto *Header = R"(
  // This is a class.
  class Foo {
  public:
    Foo();

    int Bar;
  private:
    double Number;
  };
  /// This is a function.
  char baz();
  template <typename T>
  T getT ();
  )";
  const auto TU = TestTU::withHeaderCode(Header);
  const auto Symbols = TU.headerSymbols();
  // Sanity check: there are more than 5 symbols available.
  EXPECT_GE(Symbols.size(), 5UL);
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);
  for (auto &Sym : Symbols) {
    const auto ProtobufMeessage = toProtobuf(Sym);
    const auto SymToProtobufAndBack = fromProtobuf(ProtobufMeessage, &Strings);
    EXPECT_TRUE(SymToProtobufAndBack.hasValue());
    EXPECT_EQ(toYAML(Sym), toYAML(*SymToProtobufAndBack));
  }
}

TEST(RemoteMarshallingTest, ReferenceSerialization) {
  TestTU TU;
  TU.HeaderCode = R"(
  int foo();
  int GlobalVariable = 42;
  class Foo {
  public:
    Foo();

    char Symbol = 'S';
  };
  template <typename T>
  T getT() { return T(); }
  )";
  TU.Code = R"(
  int foo() {
    ++GlobalVariable;

    Foo foo = Foo();
    if (foo.Symbol - 'a' == 42) {
      foo.Symbol = 'b';
    }

    const auto bar = getT<Foo>();
  }
  )";
  const auto References = TU.headerRefs();
  llvm::BumpPtrAllocator Arena;
  llvm::UniqueStringSaver Strings(Arena);
  // Sanity check: there are more than 5 references available.
  EXPECT_GE(References.numRefs(), 5UL);
  for (const auto &SymbolWithRefs : References) {
    for (const auto &Ref : SymbolWithRefs.second) {
      const auto RefToProtobufAndBack = fromProtobuf(toProtobuf(Ref), &Strings);
      EXPECT_TRUE(RefToProtobufAndBack.hasValue());
      EXPECT_EQ(toYAML(Ref), toYAML(*RefToProtobufAndBack));
    }
  }
} // namespace

} // namespace
} // namespace remote
} // namespace clangd
} // namespace clang
