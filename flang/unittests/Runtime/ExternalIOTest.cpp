//===-- flang/unittests/RuntimeGTest/ExternalIOTest.cpp ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sanity test for all external I/O modes
//
//===----------------------------------------------------------------------===//

#include "CrashHandlerFixture.h"
#include "gtest/gtest.h"
#include "../../runtime/io-api.h"
#include "../../runtime/main.h"
#include "../../runtime/stop.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace Fortran::runtime::io;

struct ExternalIOTests : public CrashHandlerFixture {};

TEST(ExternalIOTests, TestDirectUnformatted) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH')
  Cookie io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';

    buffer = j;
    ASSERT_TRUE(IONAME(OutputUnformattedBlock)(
        io, reinterpret_cast<const char *>(&buffer), recl, recl))
        << "OutputUnformattedBlock()";

    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputUnformattedBlock";
  }

  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(
        io, reinterpret_cast<char *>(&buffer), recl, recl))
        << "InputUnformattedBlock()";

    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputUnformattedBlock";

    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from direct unformatted record " << j
                         << ", expected " << j << '\n';
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestDirectUnformattedSwapped) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH',CONVERT='NATIVE')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";
  ASSERT_TRUE(IONAME(SetConvert)(io, "NATIVE", 6)) << "SetConvert(NATIVE)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    buffer = j;
    ASSERT_TRUE(IONAME(OutputUnformattedBlock)(
        io, reinterpret_cast<const char *>(&buffer), recl, recl))
        << "OutputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputUnformattedBlock";
  }

  // OPEN(UNIT=unit,STATUS='OLD',CONVERT='SWAP')
  io = IONAME(BeginOpenUnit)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "OLD", 3)) << "SetStatus(OLD)";
  ASSERT_TRUE(IONAME(SetConvert)(io, "SWAP", 4)) << "SetConvert(SWAP)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenUnit";

  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(
        io, reinterpret_cast<char *>(&buffer), recl, recl))
        << "InputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputUnformattedBlock";
    ASSERT_EQ(buffer >> 56, j)
        << "Read back " << (buffer >> 56) << " from direct unformatted record "
        << j << ", expected " << j << '\n';
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestSequentialFixedUnformatted) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";

  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};

  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static const int records{10};
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; WRITE(UNIT=unit) j; END DO
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    buffer = j;
    ASSERT_TRUE(IONAME(OutputUnformattedBlock)(
        io, reinterpret_cast<const char *>(&buffer), recl, recl))
        << "OutputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputUnformattedBlock";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; READ(UNIT=unit) n; check n; END DO
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(
        io, reinterpret_cast<char *>(&buffer), recl, recl))
        << "InputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputUnformattedBlock";
    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from sequential fixed unformatted record " << j
                         << ", expected " << j << '\n';
  }

  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(UNIT=unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (before read)";
    // READ(UNIT=unit) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(
        io, reinterpret_cast<char *>(&buffer), recl, recl))
        << "InputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputUnformattedBlock";
    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from sequential fixed unformatted record " << j
                         << " after backspacing, expected " << j << '\n';
    // BACKSPACE(UNIT=unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (after read)";
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestSequentialVariableUnformatted) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};

  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "UNFORMATTED", 11)) << "SetForm(UNFORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static const int records{10};
  std::int64_t buffer[records]; // INTEGER*8 :: BUFFER(0:9) = [(j,j=0,9)]
  for (int j{0}; j < records; ++j) {
    buffer[j] = j;
  }

  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; WRITE(UNIT=unit) BUFFER(0:j); END DO
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(OutputUnformattedBlock)(io,
        reinterpret_cast<const char *>(&buffer), j * sizeof *buffer,
        sizeof *buffer))
        << "OutputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputUnformattedBlock";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; READ(UNIT=unit) n; check n; END DO
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(io,
        reinterpret_cast<char *>(&buffer), j * sizeof *buffer, sizeof *buffer))
        << "InputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputUnformattedBlock";
    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], k) << "Read back [" << k << "]=" << buffer[k]
                              << " from direct unformatted record " << j
                              << ", expected " << k << '\n';
    }
  }

  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (before read)";
    // READ(unit=unit) n; check
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(InputUnformattedBlock)(io,
        reinterpret_cast<char *>(&buffer), j * sizeof *buffer, sizeof *buffer))
        << "InputUnformattedBlock()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputUnformattedBlock";
    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], k) << "Read back [" << k << "]=" << buffer[k]
                              << " from sequential variable unformatted record "
                              << j << ", expected " << k << '\n';
    }
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (after read)";
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestDirectFormatted) {
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='FORMATTED',RECL=8,STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "DIRECT", 6)) << "SetAccess(DIRECT)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";

  static constexpr std::size_t recl{8};
  ASSERT_TRUE(IONAME(SetRecl)(io, recl)) << "SetRecl()";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static constexpr int records{10};
  static const char fmt[]{"(I4)"};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,FMT=fmt,REC=j) j
    io = IONAME(BeginExternalFormattedOutput)(
        fmt, sizeof fmt - 1, unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    ASSERT_TRUE(IONAME(OutputInteger64)(io, j)) << "OutputInteger64()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputInteger64";
  }

  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,FMT=fmt,REC=j) n
    io = IONAME(BeginExternalFormattedInput)(
        fmt, sizeof fmt - 1, unit, __FILE__, __LINE__);
    ASSERT_TRUE(IONAME(SetRec)(io, j)) << "SetRec(" << j << ')';
    std::int64_t buffer;
    ASSERT_TRUE(IONAME(InputInteger)(io, buffer)) << "InputInteger()";
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputInteger";

    ASSERT_EQ(buffer, j) << "Read back " << buffer
                         << " from direct formatted record " << j
                         << ", expected " << j << '\n';
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}

TEST(ExternalIOTests, TestSequentialVariableFormatted) {
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='SCRATCH')
  auto *io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  ASSERT_TRUE(IONAME(SetAccess)(io, "SEQUENTIAL", 10))
      << "SetAccess(SEQUENTIAL)";
  ASSERT_TRUE(IONAME(SetAction)(io, "READWRITE", 9)) << "SetAction(READWRITE)";
  ASSERT_TRUE(IONAME(SetForm)(io, "FORMATTED", 9)) << "SetForm(FORMATTED)";
  ASSERT_TRUE(IONAME(SetStatus)(io, "SCRATCH", 7)) << "SetStatus(SCRATCH)";

  int unit{-1};
  ASSERT_TRUE(IONAME(GetNewUnit)(io, unit)) << "GetNewUnit()";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for OpenNewUnit";

  static const int records{10};
  std::int64_t buffer[records]; // INTEGER*8 :: BUFFER(0:9) = [(j,j=0,9)]
  for (int j{0}; j < records; ++j) {
    buffer[j] = j;
  }

  char fmt[32];
  for (int j{1}; j <= records; ++j) {
    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // DO J=1,RECORDS; WRITE(UNIT=unit,FMT=fmt) BUFFER(0:j); END DO
    io = IONAME(BeginExternalFormattedOutput)(
        fmt, std::strlen(fmt), unit, __FILE__, __LINE__);
    for (int k{0}; k < j; ++k) {
      ASSERT_TRUE(IONAME(OutputInteger64)(io, buffer[k]))
          << "OutputInteger64()";
    }
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for OutputInteger64";
  }

  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Rewind";

  for (int j{1}; j <= records; ++j) {
    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // DO J=1,RECORDS; READ(UNIT=unit,FMT=fmt) n; check n; END DO
    io = IONAME(BeginExternalFormattedInput)(
        fmt, std::strlen(fmt), unit, __FILE__, __LINE__);

    std::int64_t check[records];
    for (int k{0}; k < j; ++k) {
      ASSERT_TRUE(IONAME(InputInteger)(io, check[k])) << "InputInteger()";
    }
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputInteger";

    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], check[k])
          << "Read back [" << k << "]=" << check[k]
          << " from sequential variable formatted record " << j << ", expected "
          << buffer[k] << '\n';
    }
  }

  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (before read)";

    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // READ(UNIT=unit,FMT=fmt) n; check
    io = IONAME(BeginExternalFormattedInput)(
        fmt, std::strlen(fmt), unit, __FILE__, __LINE__);

    std::int64_t check[records];
    for (int k{0}; k < j; ++k) {
      ASSERT_TRUE(IONAME(InputInteger)(io, check[k])) << "InputInteger()";
    }

    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for InputInteger";
    for (int k{0}; k < j; ++k) {
      ASSERT_EQ(buffer[k], check[k])
          << "Read back [" << k << "]=" << buffer[k]
          << " from sequential variable formatted record " << j << ", expected "
          << buffer[k] << '\n';
    }

    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
        << "EndIoStatement() for Backspace (after read)";
  }

  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  ASSERT_TRUE(IONAME(SetStatus)(io, "DELETE", 6)) << "SetStatus(DELETE)";
  ASSERT_EQ(IONAME(EndIoStatement)(io), IostatOk)
      << "EndIoStatement() for Close";
}
