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

template <int kind = sizeof(std::int64_t)>
static OwningPtr<Descriptor> EmptyIntDescriptor() {
  OwningPtr<Descriptor> descriptor{Descriptor::Create(TypeCategory::Integer,
      kind, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  if (descriptor->Allocate() != 0) {
    return nullptr;
  }
  return descriptor;
}

class CommandFixture : public ::testing::Test {
protected:
  CommandFixture(int argc, const char *argv[]) {
    RTNAME(ProgramStart)(argc, argv, {}, {});
  }

  std::string GetPaddedStr(const char *text, std::size_t len) const {
    std::string res{text};
    assert(res.length() <= len && "No room to pad");
    res.append(len - res.length(), ' ');
    return res;
  }

  void CheckDescriptorEqStr(
      const Descriptor *value, const std::string &expected) const {
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(std::strncmp(value->OffsetElement(), expected.c_str(),
                  value->ElementBytes()),
        0)
        << "expected: " << expected << "\n"
        << "value: "
        << std::string{value->OffsetElement(), value->ElementBytes()};
  }

  template <typename INT_T = std::int64_t>
  void CheckDescriptorEqInt(
      const Descriptor *value, const INT_T expected) const {
    if (expected != -1) {
      ASSERT_NE(value, nullptr);
      EXPECT_EQ(*value->OffsetElement<INT_T>(), expected);
    }
  }

  template <typename RuntimeCall>
  void CheckValue(RuntimeCall F, const char *expectedValue,
      std::int64_t expectedLength = -1, std::int32_t expectedStatus = 0,
      const char *expectedErrMsg = "shouldn't change") const {
    OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
    ASSERT_NE(value, nullptr);

    OwningPtr<Descriptor> length{
        expectedLength == -1 ? nullptr : EmptyIntDescriptor()};

    OwningPtr<Descriptor> errmsg{CharDescriptor(expectedErrMsg)};
    ASSERT_NE(errmsg, nullptr);

    std::string expectedValueStr{
        GetPaddedStr(expectedValue, value->ElementBytes())};

    EXPECT_EQ(F(value.get(), length.get(), errmsg.get()), expectedStatus);
    CheckDescriptorEqStr(value.get(), expectedValueStr);
    CheckDescriptorEqInt(length.get(), expectedLength);
    CheckDescriptorEqStr(errmsg.get(), expectedErrMsg);
  }

  void CheckArgumentValue(const char *expectedValue, int n) const {
    SCOPED_TRACE(n);
    SCOPED_TRACE("Checking argument:");
    CheckValue(
        [&](const Descriptor *value, const Descriptor *length,
            const Descriptor *errmsg) {
          return RTNAME(GetCommandArgument)(n, value, length, errmsg);
        },
        expectedValue, std::strlen(expectedValue));
  }

  void CheckCommandValue(const char *args[], int n) const {
    SCOPED_TRACE("Checking command:");
    ASSERT_GE(n, 1);
    std::string expectedValue{args[0]};
    for (int i = 1; i < n; i++) {
      expectedValue += " " + std::string{args[i]};
    }
    CheckValue(
        [&](const Descriptor *value, const Descriptor *length,
            const Descriptor *errmsg) {
          return RTNAME(GetCommand)(value, length, errmsg);
        },
        expectedValue.c_str(), expectedValue.size());
  }

  void CheckEnvVarValue(
      const char *expectedValue, const char *name, bool trimName = true) const {
    SCOPED_TRACE(name);
    SCOPED_TRACE("Checking environment variable");
    CheckValue(
        [&](const Descriptor *value, const Descriptor *,
            const Descriptor *errmsg) {
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
        [&](const Descriptor *value, const Descriptor *,
            const Descriptor *errmsg) {
          return RTNAME(EnvVariableValue)(*nameDescriptor, value, trimName,
              errmsg, /*sourceFile=*/nullptr, /*line=*/0);
        },
        "", -1, 1, "Missing environment variable");
  }

  void CheckMissingArgumentValue(int n, const char *errStr = nullptr) const {
    OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
    ASSERT_NE(value, nullptr);

    OwningPtr<Descriptor> length{EmptyIntDescriptor()};
    ASSERT_NE(length, nullptr);

    OwningPtr<Descriptor> err{errStr ? CreateEmptyCharDescriptor() : nullptr};

    EXPECT_GT(
        RTNAME(GetCommandArgument)(n, value.get(), length.get(), err.get()), 0);

    std::string spaces(value->ElementBytes(), ' ');
    CheckDescriptorEqStr(value.get(), spaces);

    CheckDescriptorEqInt(length.get(), 0);

    if (errStr) {
      std::string paddedErrStr(GetPaddedStr(errStr, err->ElementBytes()));
      CheckDescriptorEqStr(err.get(), paddedErrStr);
    }
  }

  void CheckMissingCommandValue(const char *errStr = nullptr) const {
    OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
    ASSERT_NE(value, nullptr);

    OwningPtr<Descriptor> length{EmptyIntDescriptor()};
    ASSERT_NE(length, nullptr);

    OwningPtr<Descriptor> err{errStr ? CreateEmptyCharDescriptor() : nullptr};

    EXPECT_GT(RTNAME(GetCommand)(value.get(), length.get(), err.get()), 0);

    std::string spaces(value->ElementBytes(), ' ');
    CheckDescriptorEqStr(value.get(), spaces);

    CheckDescriptorEqInt(length.get(), 0);

    if (errStr) {
      std::string paddedErrStr(GetPaddedStr(errStr, err->ElementBytes()));
      CheckDescriptorEqStr(err.get(), paddedErrStr);
    }
  }
};

class NoArgv : public CommandFixture {
protected:
  NoArgv() : CommandFixture(0, nullptr) {}
};

// TODO: Test other intrinsics with this fixture.

TEST_F(NoArgv, GetCommand) { CheckMissingCommandValue(); }

static const char *commandOnlyArgv[]{"aProgram"};
class ZeroArguments : public CommandFixture {
protected:
  ZeroArguments() : CommandFixture(1, commandOnlyArgv) {}
};

TEST_F(ZeroArguments, ArgumentCount) { EXPECT_EQ(0, RTNAME(ArgumentCount)()); }

TEST_F(ZeroArguments, GetCommandArgument) {
  CheckMissingArgumentValue(-1);
  CheckArgumentValue(commandOnlyArgv[0], 0);
  CheckMissingArgumentValue(1);
}

TEST_F(ZeroArguments, GetCommand) { CheckCommandValue(commandOnlyArgv, 1); }

static const char *oneArgArgv[]{"aProgram", "anArgumentOfLength20"};
class OneArgument : public CommandFixture {
protected:
  OneArgument() : CommandFixture(2, oneArgArgv) {}
};

TEST_F(OneArgument, ArgumentCount) { EXPECT_EQ(1, RTNAME(ArgumentCount)()); }

TEST_F(OneArgument, GetCommandArgument) {
  CheckMissingArgumentValue(-1);
  CheckArgumentValue(oneArgArgv[0], 0);
  CheckArgumentValue(oneArgArgv[1], 1);
  CheckMissingArgumentValue(2);
}

TEST_F(OneArgument, GetCommand) { CheckCommandValue(oneArgArgv, 2); }

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

TEST_F(SeveralArguments, GetCommandArgument) {
  CheckArgumentValue(severalArgsArgv[0], 0);
  CheckArgumentValue(severalArgsArgv[1], 1);
  CheckArgumentValue(severalArgsArgv[3], 3);
  CheckArgumentValue(severalArgsArgv[4], 4);
}

TEST_F(SeveralArguments, NoArgumentValue) {
  // Make sure we don't crash if the 'value', 'length' and 'error' parameters
  // aren't passed.
  EXPECT_GT(RTNAME(GetCommandArgument)(2), 0);
  EXPECT_EQ(RTNAME(GetCommandArgument)(1), 0);
  EXPECT_GT(RTNAME(GetCommandArgument)(-1), 0);
}

TEST_F(SeveralArguments, MissingArguments) {
  CheckMissingArgumentValue(-1, "Invalid argument number");
  CheckMissingArgumentValue(2, "Missing argument");
  CheckMissingArgumentValue(5, "Invalid argument number");
  CheckMissingArgumentValue(5);
}

TEST_F(SeveralArguments, ArgValueTooShort) {
  OwningPtr<Descriptor> tooShort{CreateEmptyCharDescriptor<15>()};
  ASSERT_NE(tooShort, nullptr);
  EXPECT_EQ(RTNAME(GetCommandArgument)(1, tooShort.get()), -1);
  CheckDescriptorEqStr(tooShort.get(), severalArgsArgv[1]);

  OwningPtr<Descriptor> length{EmptyIntDescriptor()};
  ASSERT_NE(length, nullptr);
  OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor()};
  ASSERT_NE(errMsg, nullptr);

  EXPECT_EQ(
      RTNAME(GetCommandArgument)(1, tooShort.get(), length.get(), errMsg.get()),
      -1);

  CheckDescriptorEqInt(length.get(), 16);
  std::string expectedErrMsg{
      GetPaddedStr("Value too short", errMsg->ElementBytes())};
  CheckDescriptorEqStr(errMsg.get(), expectedErrMsg);
}

TEST_F(SeveralArguments, ArgErrMsgTooShort) {
  OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor<3>()};
  EXPECT_GT(RTNAME(GetCommandArgument)(-1, nullptr, nullptr, errMsg.get()), 0);
  CheckDescriptorEqStr(errMsg.get(), "Inv");
}

TEST_F(SeveralArguments, GetCommand) {
  CheckMissingCommandValue();
  CheckMissingCommandValue("Missing argument");
}

TEST_F(SeveralArguments, CommandErrMsgTooShort) {
  OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
  OwningPtr<Descriptor> length{EmptyIntDescriptor()};
  OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor<3>()};

  EXPECT_GT(RTNAME(GetCommand)(value.get(), length.get(), errMsg.get()), 0);

  std::string spaces(value->ElementBytes(), ' ');
  CheckDescriptorEqStr(value.get(), spaces);
  CheckDescriptorEqInt(length.get(), 0);
  CheckDescriptorEqStr(errMsg.get(), "Mis");
}

TEST_F(SeveralArguments, GetCommandCanTakeNull) {
  EXPECT_GT(RTNAME(GetCommand)(nullptr, nullptr, nullptr), 0);
}

static const char *onlyValidArgsArgv[]{
    "aProgram", "-f", "has/a/few/slashes", "has\\a\\few\\backslashes"};
class OnlyValidArguments : public CommandFixture {
protected:
  OnlyValidArguments()
      : CommandFixture(sizeof(onlyValidArgsArgv) / sizeof(*onlyValidArgsArgv),
            onlyValidArgsArgv) {}
};

TEST_F(OnlyValidArguments, GetCommand) {
  CheckCommandValue(onlyValidArgsArgv, 4);
}

TEST_F(OnlyValidArguments, CommandValueTooShort) {
  OwningPtr<Descriptor> tooShort{CreateEmptyCharDescriptor<50>()};
  ASSERT_NE(tooShort, nullptr);
  OwningPtr<Descriptor> length{EmptyIntDescriptor()};
  ASSERT_NE(length, nullptr);

  EXPECT_EQ(RTNAME(GetCommand)(tooShort.get(), length.get(), nullptr), -1);

  CheckDescriptorEqStr(
      tooShort.get(), "aProgram -f has/a/few/slashes has\\a\\few\\backslashe");
  CheckDescriptorEqInt(length.get(), 51);

  OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor()};
  ASSERT_NE(errMsg, nullptr);

  EXPECT_EQ(-1, RTNAME(GetCommand)(tooShort.get(), nullptr, errMsg.get()));

  std::string expectedErrMsg{
      GetPaddedStr("Value too short", errMsg->ElementBytes())};
  CheckDescriptorEqStr(errMsg.get(), expectedErrMsg);
}

TEST_F(OnlyValidArguments, GetCommandCanTakeNull) {
  EXPECT_EQ(0, RTNAME(GetCommand)(nullptr, nullptr, nullptr));

  OwningPtr<Descriptor> value{CreateEmptyCharDescriptor()};
  ASSERT_NE(value, nullptr);
  OwningPtr<Descriptor> length{EmptyIntDescriptor()};
  ASSERT_NE(length, nullptr);

  EXPECT_EQ(0, RTNAME(GetCommand)(value.get(), nullptr, nullptr));
  CheckDescriptorEqStr(value.get(),
      GetPaddedStr("aProgram -f has/a/few/slashes has\\a\\few\\backslashes",
          value->ElementBytes()));

  EXPECT_EQ(0, RTNAME(GetCommand)(nullptr, length.get(), nullptr));
  CheckDescriptorEqInt(length.get(), 51);
}

TEST_F(OnlyValidArguments, GetCommandShortLength) {
  OwningPtr<Descriptor> length{EmptyIntDescriptor<sizeof(short)>()};
  ASSERT_NE(length, nullptr);

  EXPECT_EQ(0, RTNAME(GetCommand)(nullptr, length.get(), nullptr));
  CheckDescriptorEqInt<short>(length.get(), 51);
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
