//===-- flang/unittests/Runtime/CommandTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Runtime/command.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/main.h"
#include <cstdlib>

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

static OwningPtr<Descriptor> CharDescriptor(const char *value) {
  std::size_t n{std::strlen(value)};
  OwningPtr<Descriptor> descriptor{Descriptor::Create(
      sizeof(char), n, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  if (descriptor->Allocate() != 0) {
    return nullptr;
  }
  std::memcpy(descriptor->OffsetElement(), value, n);
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
        0)
        << "expected: " << expected << "\n"
        << "value: "
        << std::string{value->OffsetElement(), value->ElementBytes()};
  }

  template <typename RuntimeCall>
  void CheckValue(RuntimeCall F, const char *expectedValue,
      std::int32_t expectedStatus = 0,
      const char *expectedErrMsg = "shouldn't change") const {
    OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
    ASSERT_NE(value, nullptr);

    OwningPtr<Descriptor> errmsg{CharDescriptor(expectedErrMsg)};

    std::string expectedValueStr{
        GetPaddedStr(expectedValue, value->ElementBytes())};

    EXPECT_EQ(F(value.get(), errmsg.get()), expectedStatus);
    CheckDescriptorEqStr(value.get(), expectedValueStr);
    CheckDescriptorEqStr(errmsg.get(), expectedErrMsg);
  }

  void CheckArgumentValue(const char *expectedValue, int n) const {
    SCOPED_TRACE(n);
    SCOPED_TRACE("Checking argument:");
    CheckValue(
        [&](const Descriptor *value, const Descriptor *errmsg) {
          return RTNAME(ArgumentValue)(n, value, errmsg);
        },
        expectedValue);
  }

  void CheckEnvVarValue(
      const char *expectedValue, const char *name, bool trimName = true) const {
    SCOPED_TRACE(name);
    SCOPED_TRACE("Checking environment variable");
    CheckValue(
        [&](const Descriptor *value, const Descriptor *errmsg) {
          return RTNAME(EnvVariableValue)(*CharDescriptor(name), value,
              trimName, errmsg, /*sourceFile=*/nullptr, /*line=*/0);
        },
        expectedValue);
  }

  void CheckMissingEnvVarValue(const char *name, bool trimName = true) const {
    SCOPED_TRACE(name);
    SCOPED_TRACE("Checking missing environment variable");

    ASSERT_EQ(nullptr, std::getenv(name))
        << "Environment variable " << name << " not expected to exist";

    OwningPtr<Descriptor> nameDescriptor{CharDescriptor(name)};
    EXPECT_EQ(0, RTNAME(EnvVariableLength)(*nameDescriptor, trimName));
    CheckValue(
        [&](const Descriptor *value, const Descriptor *errmsg) {
          return RTNAME(EnvVariableValue)(*nameDescriptor, value, trimName,
              errmsg, /*sourceFile=*/nullptr, /*line=*/0);
        },
        "", 1, "Missing environment variable");
  }

  void CheckMissingArgumentValue(int n, const char *errStr = nullptr) const {
    OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
    ASSERT_NE(value, nullptr);

    OwningPtr<Descriptor> err{errStr ? CreateEmptyCharDescriptor() : nullptr};

    EXPECT_GT(RTNAME(ArgumentValue)(n, value.get(), err.get()), 0);

    std::string spaces(value->ElementBytes(), ' ');
    CheckDescriptorEqStr(value.get(), spaces);

    if (errStr) {
      std::string paddedErrStr(GetPaddedStr(errStr, err->ElementBytes()));
      CheckDescriptorEqStr(err.get(), paddedErrStr);
    }
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
  CheckArgumentValue(commandOnlyArgv[0], 0);
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
  CheckArgumentValue(oneArgArgv[0], 0);
  CheckArgumentValue(oneArgArgv[1], 1);
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
  CheckArgumentValue(severalArgsArgv[0], 0);
  CheckArgumentValue(severalArgsArgv[1], 1);
  CheckArgumentValue(severalArgsArgv[3], 3);
  CheckArgumentValue(severalArgsArgv[4], 4);
}

TEST_F(SeveralArguments, NoArgumentValue) {
  // Make sure we don't crash if the 'value' and 'error' parameters aren't
  // passed.
  EXPECT_EQ(RTNAME(ArgumentValue)(2, nullptr, nullptr), 0);
  EXPECT_GT(RTNAME(ArgumentValue)(-1, nullptr, nullptr), 0);
}

TEST_F(SeveralArguments, MissingArguments) {
  CheckMissingArgumentValue(-1, "Invalid argument number");
  CheckMissingArgumentValue(2, "Missing argument");
  CheckMissingArgumentValue(5, "Invalid argument number");
  CheckMissingArgumentValue(5);
}

TEST_F(SeveralArguments, ValueTooShort) {
  OwningPtr<Descriptor> tooShort{CreateEmptyCharDescriptor<15>()};
  ASSERT_NE(tooShort, nullptr);
  EXPECT_EQ(RTNAME(ArgumentValue)(1, tooShort.get(), nullptr), -1);
  CheckDescriptorEqStr(tooShort.get(), severalArgsArgv[1]);

  OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor()};
  ASSERT_NE(errMsg, nullptr);

  EXPECT_EQ(RTNAME(ArgumentValue)(1, tooShort.get(), errMsg.get()), -1);

  std::string expectedErrMsg{
      GetPaddedStr("Value too short", errMsg->ElementBytes())};
  CheckDescriptorEqStr(errMsg.get(), expectedErrMsg);
}

TEST_F(SeveralArguments, ErrMsgTooShort) {
  OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor<3>()};
  EXPECT_GT(RTNAME(ArgumentValue)(-1, nullptr, errMsg.get()), 0);
  CheckDescriptorEqStr(errMsg.get(), "Inv");
}

class EnvironmentVariables : public CommandFixture {
protected:
  EnvironmentVariables() : CommandFixture(0, nullptr) {
    SetEnv("NAME", "VALUE");
    SetEnv("EMPTY", "");
  }

  // If we have access to setenv, we can run some more fine-grained tests.
  template <typename ParamType = char>
  void SetEnv(const ParamType *name, const ParamType *value,
      decltype(setenv(name, value, 1)) *Enabled = nullptr) {
    ASSERT_EQ(0, setenv(name, value, /*overwrite=*/1));
    canSetEnv = true;
  }

  // Fallback method if setenv is not available.
  template <typename Unused = void> void SetEnv(const void *, const void *) {}

  bool EnableFineGrainedTests() const { return canSetEnv; }

private:
  bool canSetEnv{false};
};

TEST_F(EnvironmentVariables, Nonexistent) {
  CheckMissingEnvVarValue("DOESNT_EXIST");

  CheckMissingEnvVarValue("      ");
  CheckMissingEnvVarValue("");
}

TEST_F(EnvironmentVariables, Basic) {
  // Test a variable that's expected to exist in the environment.
  char *path{std::getenv("PATH")};
  auto expectedLen{static_cast<int64_t>(std::strlen(path))};
  EXPECT_EQ(expectedLen, RTNAME(EnvVariableLength)(*CharDescriptor("PATH")));
}

TEST_F(EnvironmentVariables, Trim) {
  if (EnableFineGrainedTests()) {
    EXPECT_EQ(5, RTNAME(EnvVariableLength)(*CharDescriptor("NAME   ")));
    CheckEnvVarValue("VALUE", "NAME   ");
  }
}

TEST_F(EnvironmentVariables, NoTrim) {
  if (EnableFineGrainedTests()) {
    CheckMissingEnvVarValue("NAME      ", /*trim_name=*/false);
  }
}

TEST_F(EnvironmentVariables, Empty) {
  if (EnableFineGrainedTests()) {
    EXPECT_EQ(0, RTNAME(EnvVariableLength)(*CharDescriptor("EMPTY")));
    CheckEnvVarValue("", "EMPTY");
  }
}

TEST_F(EnvironmentVariables, NoValueOrErrmsg) {
  ASSERT_EQ(std::getenv("DOESNT_EXIST"), nullptr)
      << "Environment variable DOESNT_EXIST actually exists";
  EXPECT_EQ(RTNAME(EnvVariableValue)(*CharDescriptor("DOESNT_EXIST")), 1);

  if (EnableFineGrainedTests()) {
    EXPECT_EQ(RTNAME(EnvVariableValue)(*CharDescriptor("NAME")), 0);
  }
}

TEST_F(EnvironmentVariables, ValueTooShort) {
  if (EnableFineGrainedTests()) {
    OwningPtr<Descriptor> tooShort{CreateEmptyCharDescriptor<2>()};
    ASSERT_NE(tooShort, nullptr);
    EXPECT_EQ(RTNAME(EnvVariableValue)(*CharDescriptor("NAME"), tooShort.get(),
                  /*trim_name=*/true, nullptr),
        -1);
    CheckDescriptorEqStr(tooShort.get(), "VALUE");

    OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor()};
    ASSERT_NE(errMsg, nullptr);

    EXPECT_EQ(RTNAME(EnvVariableValue)(*CharDescriptor("NAME"), tooShort.get(),
                  /*trim_name=*/true, errMsg.get()),
        -1);

    std::string expectedErrMsg{
        GetPaddedStr("Value too short", errMsg->ElementBytes())};
    CheckDescriptorEqStr(errMsg.get(), expectedErrMsg);
  }
}

TEST_F(EnvironmentVariables, ErrMsgTooShort) {
  ASSERT_EQ(std::getenv("DOESNT_EXIST"), nullptr)
      << "Environment variable DOESNT_EXIST actually exists";

  OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor<3>()};
  EXPECT_EQ(RTNAME(EnvVariableValue)(*CharDescriptor("DOESNT_EXIST"), nullptr,
                /*trim_name=*/true, errMsg.get()),
      1);
  CheckDescriptorEqStr(errMsg.get(), "Mis");
}
