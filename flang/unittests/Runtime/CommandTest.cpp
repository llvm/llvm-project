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
#include "flang/Runtime/execute.h"
#include "flang/Runtime/extensions.h"
#include "flang/Runtime/main.h"
#include <cstddef>
#include <cstdlib>

#if _REENTRANT || _POSIX_C_SOURCE >= 199506L
#include <limits.h> // LOGIN_NAME_MAX used in getlog test
#endif

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

template <int kind = sizeof(std::int64_t)>
static OwningPtr<Descriptor> IntDescriptor(const int &value) {
  OwningPtr<Descriptor> descriptor{Descriptor::Create(TypeCategory::Integer,
      kind, nullptr, 0, nullptr, CFI_attribute_allocatable)};
  if (descriptor->Allocate() != 0) {
    return nullptr;
  }
  std::memcpy(descriptor->OffsetElement<int>(), &value, sizeof(int));
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

  void CheckCharEqStr(const char *value, const std::string &expected) const {
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(std::strncmp(value, expected.c_str(), expected.size()), 0)
        << "expected: " << expected << "\n"
        << "value: " << value;
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
        [&](const Descriptor *value, const Descriptor *length,
            const Descriptor *errmsg) {
          return RTNAME(GetEnvVariable)(
              *CharDescriptor(name), value, length, trimName, errmsg);
        },
        expectedValue, std::strlen(expectedValue));
  }

  void CheckMissingEnvVarValue(const char *name, bool trimName = true) const {
    SCOPED_TRACE(name);
    SCOPED_TRACE("Checking missing environment variable");

    ASSERT_EQ(nullptr, std::getenv(name))
        << "Environment variable " << name << " not expected to exist";

    CheckValue(
        [&](const Descriptor *value, const Descriptor *length,
            const Descriptor *errmsg) {
          return RTNAME(GetEnvVariable)(
              *CharDescriptor(name), value, length, trimName, errmsg);
        },
        "", 0, 1, "Missing environment variable");
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

    CheckDescriptorEqInt<std::int64_t>(length.get(), 0);

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

    CheckDescriptorEqInt<std::int64_t>(length.get(), 0);

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

#if _WIN32 || _POSIX_C_SOURCE >= 1 || _XOPEN_SOURCE || _BSD_SOURCE || \
    _SVID_SOURCE || defined(_POSIX_SOURCE)
TEST_F(NoArgv, FdateGetDate) {
  char input[]{"24LengthCharIsJustRight"};
  const std::size_t charLen = sizeof(input);

  FORTRAN_PROCEDURE_NAME(fdate)(input, charLen);

  // Tue May 26 21:51:03 2015\n\0
  // index at 3, 7, 10, 19 should be space
  // when date is less than two digit, index 8 would be space
  // Tue May  6 21:51:03 2015\n\0
  for (std::size_t i{0}; i < charLen; i++) {
    if (i == 8)
      continue;
    if (i == 3 || i == 7 || i == 10 || i == 19) {
      EXPECT_EQ(input[i], ' ');
      continue;
    }
    EXPECT_NE(input[i], ' ');
  }
}

TEST_F(NoArgv, FdateGetDateTooShort) {
  char input[]{"TooShortAllPadSpace"};
  const std::size_t charLen = sizeof(input);

  FORTRAN_PROCEDURE_NAME(fdate)(input, charLen);

  for (std::size_t i{0}; i < charLen; i++) {
    EXPECT_EQ(input[i], ' ');
  }
}

TEST_F(NoArgv, FdateGetDatePadSpace) {
  char input[]{"All char after 23 pad spaces"};
  const std::size_t charLen = sizeof(input);

  FORTRAN_PROCEDURE_NAME(fdate)(input, charLen);

  for (std::size_t i{24}; i < charLen; i++) {
    EXPECT_EQ(input[i], ' ');
  }
}

#else
TEST_F(NoArgv, FdateNotSupported) {
  char input[]{"No change due to crash"};

  EXPECT_DEATH(FORTRAN_PROCEDURE_NAME(fdate)(input, sizeof(input)),
      "fdate is not supported.");

  CheckCharEqStr(input, "No change due to crash");
}
#endif

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

TEST_F(ZeroArguments, ECLValidCommandAndPadSync) {
  OwningPtr<Descriptor> command{CharDescriptor("echo hi")};
  bool wait{true};
  OwningPtr<Descriptor> exitStat{EmptyIntDescriptor()};
  OwningPtr<Descriptor> cmdStat{EmptyIntDescriptor()};
  OwningPtr<Descriptor> cmdMsg{CharDescriptor("No change")};

  RTNAME(ExecuteCommandLine)
  (*command.get(), wait, exitStat.get(), cmdStat.get(), cmdMsg.get());

  std::string spaces(cmdMsg->ElementBytes(), ' ');
  CheckDescriptorEqInt<std::int64_t>(exitStat.get(), 0);
  CheckDescriptorEqInt<std::int64_t>(cmdStat.get(), 0);
  CheckDescriptorEqStr(cmdMsg.get(), "No change");
}

TEST_F(ZeroArguments, ECLValidCommandStatusSetSync) {
  OwningPtr<Descriptor> command{CharDescriptor("echo hi")};
  bool wait{true};
  OwningPtr<Descriptor> exitStat{IntDescriptor(404)};
  OwningPtr<Descriptor> cmdStat{IntDescriptor(202)};
  OwningPtr<Descriptor> cmdMsg{CharDescriptor("No change")};

  RTNAME(ExecuteCommandLine)
  (*command.get(), wait, exitStat.get(), cmdStat.get(), cmdMsg.get());

  CheckDescriptorEqInt<std::int64_t>(exitStat.get(), 0);
  CheckDescriptorEqInt<std::int64_t>(cmdStat.get(), 0);
  CheckDescriptorEqStr(cmdMsg.get(), "No change");
}

TEST_F(ZeroArguments, ECLInvalidCommandErrorSync) {
  OwningPtr<Descriptor> command{CharDescriptor("InvalidCommand")};
  bool wait{true};
  OwningPtr<Descriptor> exitStat{IntDescriptor(404)};
  OwningPtr<Descriptor> cmdStat{IntDescriptor(202)};
  OwningPtr<Descriptor> cmdMsg{CharDescriptor("Message ChangedXXXXXXXXX")};

  RTNAME(ExecuteCommandLine)
  (*command.get(), wait, exitStat.get(), cmdStat.get(), cmdMsg.get());
#ifdef _WIN32
  CheckDescriptorEqInt(exitStat.get(), 1);
#else
  CheckDescriptorEqInt<std::int64_t>(exitStat.get(), 127);
#endif
  CheckDescriptorEqInt<std::int64_t>(cmdStat.get(), 3);
  CheckDescriptorEqStr(cmdMsg.get(), "Invalid command lineXXXX");
}

TEST_F(ZeroArguments, ECLInvalidCommandTerminatedSync) {
  OwningPtr<Descriptor> command{CharDescriptor("InvalidCommand")};
  bool wait{true};
  OwningPtr<Descriptor> exitStat{IntDescriptor(404)};
  OwningPtr<Descriptor> cmdMsg{CharDescriptor("No Change")};

#ifdef _WIN32
  EXPECT_DEATH(RTNAME(ExecuteCommandLine)(
                   *command.get(), wait, exitStat.get(), nullptr, cmdMsg.get()),
      "Invalid command quit with exit status code: 1");
#else
  EXPECT_DEATH(RTNAME(ExecuteCommandLine)(
                   *command.get(), wait, exitStat.get(), nullptr, cmdMsg.get()),
      "Invalid command quit with exit status code: 127");
#endif
  CheckDescriptorEqInt(exitStat.get(), 404);
  CheckDescriptorEqStr(cmdMsg.get(), "No Change");
}

TEST_F(ZeroArguments, ECLValidCommandAndExitStatNoChangeAndCMDStatusSetAsync) {
  OwningPtr<Descriptor> command{CharDescriptor("echo hi")};
  bool wait{false};
  OwningPtr<Descriptor> exitStat{IntDescriptor(404)};
  OwningPtr<Descriptor> cmdStat{IntDescriptor(202)};
  OwningPtr<Descriptor> cmdMsg{CharDescriptor("No change")};

  RTNAME(ExecuteCommandLine)
  (*command.get(), wait, exitStat.get(), cmdStat.get(), cmdMsg.get());

  CheckDescriptorEqInt(exitStat.get(), 404);
  CheckDescriptorEqInt<std::int64_t>(cmdStat.get(), 0);
  CheckDescriptorEqStr(cmdMsg.get(), "No change");
}

TEST_F(ZeroArguments, ECLInvalidCommandParentNotTerminatedAsync) {
  OwningPtr<Descriptor> command{CharDescriptor("InvalidCommand")};
  bool wait{false};
  OwningPtr<Descriptor> exitStat{IntDescriptor(404)};
  OwningPtr<Descriptor> cmdMsg{CharDescriptor("No change")};

  EXPECT_NO_FATAL_FAILURE(RTNAME(ExecuteCommandLine)(
      *command.get(), wait, exitStat.get(), nullptr, cmdMsg.get()));

  CheckDescriptorEqInt(exitStat.get(), 404);
  CheckDescriptorEqStr(cmdMsg.get(), "No change");
}

TEST_F(ZeroArguments, ECLInvalidCommandAsyncDontAffectSync) {
  OwningPtr<Descriptor> command{CharDescriptor("echo hi")};

  EXPECT_NO_FATAL_FAILURE(RTNAME(ExecuteCommandLine)(
      *command.get(), false, nullptr, nullptr, nullptr));
  EXPECT_NO_FATAL_FAILURE(RTNAME(ExecuteCommandLine)(
      *command.get(), true, nullptr, nullptr, nullptr));
}

TEST_F(ZeroArguments, ECLInvalidCommandAsyncDontAffectAsync) {
  OwningPtr<Descriptor> command{CharDescriptor("echo hi")};

  EXPECT_NO_FATAL_FAILURE(RTNAME(ExecuteCommandLine)(
      *command.get(), false, nullptr, nullptr, nullptr));
  EXPECT_NO_FATAL_FAILURE(RTNAME(ExecuteCommandLine)(
      *command.get(), false, nullptr, nullptr, nullptr));
}

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

  CheckDescriptorEqInt<std::int64_t>(length.get(), 16);
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
  CheckDescriptorEqInt<std::int64_t>(length.get(), 0);
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
  CheckDescriptorEqInt<std::int64_t>(length.get(), 51);

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
  CheckDescriptorEqInt<std::int64_t>(length.get(), 51);
}

TEST_F(OnlyValidArguments, GetCommandShortLength) {
  OwningPtr<Descriptor> length{EmptyIntDescriptor<sizeof(short)>()};
  ASSERT_NE(length, nullptr);

  EXPECT_EQ(0, RTNAME(GetCommand)(nullptr, length.get(), nullptr));
  CheckDescriptorEqInt<short>(length.get(), 51);
}

TEST_F(ZeroArguments, GetPID) {
  // pid should always greater than 0, in both linux and windows
  EXPECT_GT(RTNAME(GetPID)(), 0);
}

class EnvironmentVariables : public CommandFixture {
protected:
  EnvironmentVariables() : CommandFixture(0, nullptr) {
    SetEnv("NAME", "VALUE");
#ifdef _WIN32
    SetEnv("USERNAME", "loginName");
#else
    SetEnv("LOGNAME", "loginName");
#endif
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
  OwningPtr<Descriptor> length{EmptyIntDescriptor()};
  EXPECT_EQ(0,
      RTNAME(GetEnvVariable)(*CharDescriptor("PATH"),
          /*value=*/nullptr, length.get()));
  CheckDescriptorEqInt(length.get(), expectedLen);
}

TEST_F(EnvironmentVariables, Trim) {
  if (EnableFineGrainedTests()) {
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
    CheckEnvVarValue("", "EMPTY");
  }
}

TEST_F(EnvironmentVariables, NoValueOrErrmsg) {
  ASSERT_EQ(std::getenv("DOESNT_EXIST"), nullptr)
      << "Environment variable DOESNT_EXIST actually exists";
  EXPECT_EQ(RTNAME(GetEnvVariable)(*CharDescriptor("DOESNT_EXIST")), 1);

  if (EnableFineGrainedTests()) {
    EXPECT_EQ(RTNAME(GetEnvVariable)(*CharDescriptor("NAME")), 0);
  }
}

TEST_F(EnvironmentVariables, ValueTooShort) {
  if (EnableFineGrainedTests()) {
    OwningPtr<Descriptor> tooShort{CreateEmptyCharDescriptor<2>()};
    ASSERT_NE(tooShort, nullptr);
    EXPECT_EQ(RTNAME(GetEnvVariable)(*CharDescriptor("NAME"), tooShort.get(),
                  /*length=*/nullptr, /*trim_name=*/true, nullptr),
        -1);
    CheckDescriptorEqStr(tooShort.get(), "VALUE");

    OwningPtr<Descriptor> errMsg{CreateEmptyCharDescriptor()};
    ASSERT_NE(errMsg, nullptr);

    EXPECT_EQ(RTNAME(GetEnvVariable)(*CharDescriptor("NAME"), tooShort.get(),
                  /*length=*/nullptr, /*trim_name=*/true, errMsg.get()),
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
  EXPECT_EQ(RTNAME(GetEnvVariable)(*CharDescriptor("DOESNT_EXIST"), nullptr,
                /*length=*/nullptr, /*trim_name=*/true, errMsg.get()),
      1);
  CheckDescriptorEqStr(errMsg.get(), "Mis");
}

// username first char must not be null
TEST_F(EnvironmentVariables, GetlogGetName) {
  const int charLen{3};
  char input[charLen]{"\0\0"};
  FORTRAN_PROCEDURE_NAME(getlog)(input, charLen);
  EXPECT_NE(input[0], '\0');
}

#if _REENTRANT || _POSIX_C_SOURCE >= 199506L
TEST_F(EnvironmentVariables, GetlogPadSpace) {
  // guarantee 1 char longer than max, last char should be pad space
  int charLen;
#ifdef LOGIN_NAME_MAX
  charLen = LOGIN_NAME_MAX + 2;
#else
  charLen = sysconf(_SC_LOGIN_NAME_MAX) + 2;
  if (charLen == -1)
    charLen = _POSIX_LOGIN_NAME_MAX + 2;
#endif
  std::vector<char> input(charLen);
  FORTRAN_PROCEDURE_NAME(getlog)(input.data(), charLen);
  EXPECT_EQ(input[charLen - 1], ' ');
}
#endif

#ifdef _WIN32 // Test ability to get name from environment variable
TEST_F(EnvironmentVariables, GetlogEnvGetName) {
  if (EnableFineGrainedTests()) {
    ASSERT_NE(std::getenv("USERNAME"), nullptr)
        << "Environment variable USERNAME does not exist";

    char input[]{"XXXXXXXXX"};
    FORTRAN_PROCEDURE_NAME(getlog)(input, sizeof(input));

    CheckCharEqStr(input, "loginName");
  }
}

TEST_F(EnvironmentVariables, GetlogEnvBufferShort) {
  if (EnableFineGrainedTests()) {
    ASSERT_NE(std::getenv("USERNAME"), nullptr)
        << "Environment variable USERNAME does not exist";

    char input[]{"XXXXXX"};
    FORTRAN_PROCEDURE_NAME(getlog)(input, sizeof(input));

    CheckCharEqStr(input, "loginN");
  }
}

TEST_F(EnvironmentVariables, GetlogEnvPadSpace) {
  if (EnableFineGrainedTests()) {
    ASSERT_NE(std::getenv("USERNAME"), nullptr)
        << "Environment variable USERNAME does not exist";

    char input[]{"XXXXXXXXXX"};
    FORTRAN_PROCEDURE_NAME(getlog)(input, sizeof(input));

    CheckCharEqStr(input, "loginName ");
  }
}
#endif
