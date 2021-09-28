//===-- flang/unittests/RuntimeGTest/CommandTest.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/command.h"
#include "gtest/gtest.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/main.h"

using namespace Fortran::runtime;

template <std::size_t n = 64>
static OwningPtr<Descriptor> CreateEmptyCharDescriptor() {
  OwningPtr<Descriptor> descriptor{Descriptor::Create(
      sizeof(char), n, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  if (descriptor->Allocate() != 0) {
    return nullptr;
  }
  return descriptor;
}

class CommandFixture : public ::testing::Test {
protected:
  CommandFixture(int argc, const char *argv[]) {
    RTNAME(ProgramStart)(argc, argv, {});
  }

  std::string GetPaddedStr(const char *text, std::size_t len) const {
    std::string res{text};
    assert(res.length() <= len && "No room to pad");
    res.append(len - res.length(), ' ');
    return res;
  }

  void CheckDescriptorEqStr(
      const Descriptor *value, const std::string &expected) const {
    EXPECT_EQ(std::strncmp(value->OffsetElement(), expected.c_str(),
                  value->ElementBytes()),
        0);
  }

  void CheckArgumentValue(int n, const char *argv) const {
    OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
    ASSERT_NE(value, nullptr);

    std::string expected{GetPaddedStr(argv, value->ElementBytes())};

    EXPECT_EQ(RTNAME(ArgumentValue)(n, value.get(), nullptr), 0);
    CheckDescriptorEqStr(value.get(), expected);
  }

  void CheckMissingArgumentValue(int n) const {
    OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
    ASSERT_NE(value, nullptr);

    EXPECT_GT(RTNAME(ArgumentValue)(n, value.get(), nullptr), 0);

    std::string spaces(value->ElementBytes(), ' ');
    CheckDescriptorEqStr(value.get(), spaces);
  }
};

static const char *commandOnlyArgv[]{"aProgram"};
class ZeroArguments : public CommandFixture {
protected:
  ZeroArguments() : CommandFixture(1, commandOnlyArgv) {}
};

TEST_F(ZeroArguments, ArgumentCount) { EXPECT_EQ(0, RTNAME(ArgumentCount)()); }

TEST_F(ZeroArguments, ArgumentLength) {
  EXPECT_EQ(0, RTNAME(ArgumentLength)(-1));
  EXPECT_EQ(8, RTNAME(ArgumentLength)(0));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(1));
}

TEST_F(ZeroArguments, ArgumentValue) {
  CheckArgumentValue(0, commandOnlyArgv[0]);
}

static const char *oneArgArgv[]{"aProgram", "anArgumentOfLength20"};
class OneArgument : public CommandFixture {
protected:
  OneArgument() : CommandFixture(2, oneArgArgv) {}
};

TEST_F(OneArgument, ArgumentCount) { EXPECT_EQ(1, RTNAME(ArgumentCount)()); }

TEST_F(OneArgument, ArgumentLength) {
  EXPECT_EQ(0, RTNAME(ArgumentLength)(-1));
  EXPECT_EQ(8, RTNAME(ArgumentLength)(0));
  EXPECT_EQ(20, RTNAME(ArgumentLength)(1));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(2));
}

TEST_F(OneArgument, ArgumentValue) {
  CheckArgumentValue(0, oneArgArgv[0]);
  CheckArgumentValue(1, oneArgArgv[1]);
}

static const char *severalArgsArgv[]{
    "aProgram", "16-char-long-arg", "", "-22-character-long-arg", "o"};
class SeveralArguments : public CommandFixture {
protected:
  SeveralArguments()
      : CommandFixture(sizeof(severalArgsArgv) / sizeof(*severalArgsArgv),
            severalArgsArgv) {}
};

TEST_F(SeveralArguments, ArgumentCount) {
  EXPECT_EQ(4, RTNAME(ArgumentCount)());
}

TEST_F(SeveralArguments, ArgumentLength) {
  EXPECT_EQ(0, RTNAME(ArgumentLength)(-1));
  EXPECT_EQ(8, RTNAME(ArgumentLength)(0));
  EXPECT_EQ(16, RTNAME(ArgumentLength)(1));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(2));
  EXPECT_EQ(22, RTNAME(ArgumentLength)(3));
  EXPECT_EQ(1, RTNAME(ArgumentLength)(4));
  EXPECT_EQ(0, RTNAME(ArgumentLength)(5));
}

TEST_F(SeveralArguments, ArgumentValue) {
  CheckArgumentValue(0, severalArgsArgv[0]);
  CheckArgumentValue(1, severalArgsArgv[1]);
  CheckMissingArgumentValue(2);
  CheckArgumentValue(3, severalArgsArgv[3]);
  CheckArgumentValue(4, severalArgsArgv[4]);
}

TEST_F(SeveralArguments, NoArgumentValue) {
  // Make sure we don't crash if the 'value' and 'error' parameters aren't
  // passed.
  EXPECT_EQ(RTNAME(ArgumentValue)(2, nullptr, nullptr), 0);
  EXPECT_GT(RTNAME(ArgumentValue)(-1, nullptr, nullptr), 0);
}

TEST_F(SeveralArguments, ArgumentValueErrors) {
  CheckMissingArgumentValue(-1);
  CheckMissingArgumentValue(5);

  OwningPtr<Descriptor> tooShort{CreateEmptyCharDescriptor<15>()};
  ASSERT_NE(tooShort, nullptr);
  EXPECT_EQ(RTNAME(ArgumentValue)(1, tooShort.get(), nullptr), -1);
  CheckDescriptorEqStr(tooShort.get(), severalArgsArgv[1]);
}
