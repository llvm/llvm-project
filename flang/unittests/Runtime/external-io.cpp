// Sanity test for all external I/O modes

#include "testing.h"
#include "../../runtime/io-api.h"
#include "../../runtime/main.h"
#include "../../runtime/stop.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>

using namespace Fortran::runtime::io;

void TestDirectUnformatted() {
  llvm::errs() << "begin TestDirectUnformatted()\n";
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH')
  auto io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(SetAccess)(io, "DIRECT", 6) || (Fail() << "SetAccess(DIRECT)", 0);
  IONAME(SetAction)
  (io, "READWRITE", 9) || (Fail() << "SetAction(READWRITE)", 0);
  IONAME(SetForm)
  (io, "UNFORMATTED", 11) || (Fail() << "SetForm(UNFORMATTED)", 0);
  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  IONAME(SetRecl)(io, recl) || (Fail() << "SetRecl()", 0);
  IONAME(SetStatus)(io, "SCRATCH", 7) || (Fail() << "SetStatus(SCRATCH)", 0);
  int unit{-1};
  IONAME(GetNewUnit)(io, unit) || (Fail() << "GetNewUnit()", 0);
  llvm::errs() << "unit=" << unit << '\n';
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for OpenNewUnit", 0);
  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    IONAME(SetRec)(io, j) || (Fail() << "SetRec(" << j << ')', 0);
    buffer = j;
    IONAME(OutputUnformattedBlock)
    (io, reinterpret_cast<const char *>(&buffer), recl, recl) ||
        (Fail() << "OutputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for OutputUnformattedBlock", 0);
  }
  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    IONAME(SetRec)
    (io, j) || (Fail() << "SetRec(" << j << ')', 0);
    IONAME(InputUnformattedBlock)
    (io, reinterpret_cast<char *>(&buffer), recl, recl) ||
        (Fail() << "InputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for InputUnformattedBlock", 0);
    if (buffer != j) {
      Fail() << "Read back " << buffer << " from direct unformatted record "
             << j << ", expected " << j << '\n';
    }
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  IONAME(SetStatus)(io, "DELETE", 6) || (Fail() << "SetStatus(DELETE)", 0);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Close", 0);
  llvm::errs() << "end TestDirectUnformatted()\n";
}

void TestDirectUnformattedSwapped() {
  llvm::errs() << "begin TestDirectUnformattedSwapped()\n";
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH',CONVERT='NATIVE')
  auto io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(SetAccess)(io, "DIRECT", 6) || (Fail() << "SetAccess(DIRECT)", 0);
  IONAME(SetAction)
  (io, "READWRITE", 9) || (Fail() << "SetAction(READWRITE)", 0);
  IONAME(SetForm)
  (io, "UNFORMATTED", 11) || (Fail() << "SetForm(UNFORMATTED)", 0);
  IONAME(SetConvert)
  (io, "NATIVE", 6) || (Fail() << "SetConvert(NATIVE)", 0);
  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  IONAME(SetRecl)(io, recl) || (Fail() << "SetRecl()", 0);
  IONAME(SetStatus)(io, "SCRATCH", 7) || (Fail() << "SetStatus(SCRATCH)", 0);
  int unit{-1};
  IONAME(GetNewUnit)(io, unit) || (Fail() << "GetNewUnit()", 0);
  llvm::errs() << "unit=" << unit << '\n';
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for OpenNewUnit", 0);
  static constexpr int records{10};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,REC=j) j
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    IONAME(SetRec)(io, j) || (Fail() << "SetRec(" << j << ')', 0);
    buffer = j;
    IONAME(OutputUnformattedBlock)
    (io, reinterpret_cast<const char *>(&buffer), recl, recl) ||
        (Fail() << "OutputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for OutputUnformattedBlock", 0);
  }
  // OPEN(UNIT=unit,STATUS='OLD',CONVERT='SWAP')
  io = IONAME(BeginOpenUnit)(unit, __FILE__, __LINE__);
  IONAME(SetStatus)(io, "OLD", 3) || (Fail() << "SetStatus(OLD)", 0);
  IONAME(SetConvert)
  (io, "SWAP", 4) || (Fail() << "SetConvert(SWAP)", 0);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for OpenUnit", 0);
  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,REC=j) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    IONAME(SetRec)
    (io, j) || (Fail() << "SetRec(" << j << ')', 0);
    IONAME(InputUnformattedBlock)
    (io, reinterpret_cast<char *>(&buffer), recl, recl) ||
        (Fail() << "InputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for InputUnformattedBlock", 0);
    if (buffer >> 56 != j) {
      Fail() << "Read back " << (buffer >> 56)
             << " from direct unformatted record " << j << ", expected " << j
             << '\n';
    }
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  IONAME(SetStatus)(io, "DELETE", 6) || (Fail() << "SetStatus(DELETE)", 0);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Close", 0);
  llvm::errs() << "end TestDirectUnformatted()\n";
}

void TestSequentialFixedUnformatted() {
  llvm::errs() << "begin TestSequentialFixedUnformatted()\n";
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',RECL=8,STATUS='SCRATCH')
  auto io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(SetAccess)
  (io, "SEQUENTIAL", 10) || (Fail() << "SetAccess(SEQUENTIAL)", 0);
  IONAME(SetAction)
  (io, "READWRITE", 9) || (Fail() << "SetAction(READWRITE)", 0);
  IONAME(SetForm)
  (io, "UNFORMATTED", 11) || (Fail() << "SetForm(UNFORMATTED)", 0);
  std::int64_t buffer;
  static constexpr std::size_t recl{sizeof buffer};
  IONAME(SetRecl)(io, recl) || (Fail() << "SetRecl()", 0);
  IONAME(SetStatus)(io, "SCRATCH", 7) || (Fail() << "SetStatus(SCRATCH)", 0);
  int unit{-1};
  IONAME(GetNewUnit)(io, unit) || (Fail() << "GetNewUnit()", 0);
  llvm::errs() << "unit=" << unit << '\n';
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for OpenNewUnit", 0);
  static const int records{10};
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; WRITE(UNIT=unit) j; END DO
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    buffer = j;
    IONAME(OutputUnformattedBlock)
    (io, reinterpret_cast<const char *>(&buffer), recl, recl) ||
        (Fail() << "OutputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for OutputUnformattedBlock", 0);
  }
  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Rewind", 0);
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; READ(UNIT=unit) n; check n; END DO
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    IONAME(InputUnformattedBlock)
    (io, reinterpret_cast<char *>(&buffer), recl, recl) ||
        (Fail() << "InputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for InputUnformattedBlock", 0);
    if (buffer != j) {
      Fail() << "Read back " << buffer
             << " from sequential fixed unformatted record " << j
             << ", expected " << j << '\n';
    }
  }
  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(UNIT=unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for Backspace (before read)", 0);
    // READ(UNIT=unit) n
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    IONAME(InputUnformattedBlock)
    (io, reinterpret_cast<char *>(&buffer), recl, recl) ||
        (Fail() << "InputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for InputUnformattedBlock", 0);
    if (buffer != j) {
      Fail() << "Read back " << buffer
             << " from sequential fixed unformatted record " << j
             << " after backspacing, expected " << j << '\n';
    }
    // BACKSPACE(UNIT=unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for Backspace (after read)", 0);
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  IONAME(SetStatus)(io, "DELETE", 6) || (Fail() << "SetStatus(DELETE)", 0);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Close", 0);
  llvm::errs() << "end TestSequentialFixedUnformatted()\n";
}

void TestSequentialVariableUnformatted() {
  llvm::errs() << "begin TestSequentialVariableUnformatted()\n";
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='UNFORMATTED',STATUS='SCRATCH')
  auto io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(SetAccess)
  (io, "SEQUENTIAL", 10) || (Fail() << "SetAccess(SEQUENTIAL)", 0);
  IONAME(SetAction)
  (io, "READWRITE", 9) || (Fail() << "SetAction(READWRITE)", 0);
  IONAME(SetForm)
  (io, "UNFORMATTED", 11) || (Fail() << "SetForm(UNFORMATTED)", 0);
  IONAME(SetStatus)(io, "SCRATCH", 7) || (Fail() << "SetStatus(SCRATCH)", 0);
  int unit{-1};
  IONAME(GetNewUnit)(io, unit) || (Fail() << "GetNewUnit()", 0);
  llvm::errs() << "unit=" << unit << '\n';
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for OpenNewUnit", 0);
  static const int records{10};
  std::int64_t buffer[records]; // INTEGER*8 :: BUFFER(0:9) = [(j,j=0,9)]
  for (int j{0}; j < records; ++j) {
    buffer[j] = j;
  }
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; WRITE(UNIT=unit) BUFFER(0:j); END DO
    io = IONAME(BeginUnformattedOutput)(unit, __FILE__, __LINE__);
    IONAME(OutputUnformattedBlock)
    (io, reinterpret_cast<const char *>(&buffer), j * sizeof *buffer,
        sizeof *buffer) ||
        (Fail() << "OutputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for OutputUnformattedBlock", 0);
  }
  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Rewind", 0);
  for (int j{1}; j <= records; ++j) {
    // DO J=1,RECORDS; READ(UNIT=unit) n; check n; END DO
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    IONAME(InputUnformattedBlock)
    (io, reinterpret_cast<char *>(&buffer), j * sizeof *buffer,
        sizeof *buffer) ||
        (Fail() << "InputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for InputUnformattedBlock", 0);
    for (int k{0}; k < j; ++k) {
      if (buffer[k] != k) {
        Fail() << "Read back [" << k << "]=" << buffer[k]
               << " from direct unformatted record " << j << ", expected " << k
               << '\n';
      }
    }
  }
  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for Backspace (before read)", 0);
    // READ(unit=unit) n; check
    io = IONAME(BeginUnformattedInput)(unit, __FILE__, __LINE__);
    IONAME(InputUnformattedBlock)
    (io, reinterpret_cast<char *>(&buffer), j * sizeof *buffer,
        sizeof *buffer) ||
        (Fail() << "InputUnformattedBlock()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for InputUnformattedBlock", 0);
    for (int k{0}; k < j; ++k) {
      if (buffer[k] != k) {
        Fail() << "Read back [" << k << "]=" << buffer[k]
               << " from sequential variable unformatted record " << j
               << ", expected " << k << '\n';
      }
    }
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for Backspace (after read)", 0);
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  IONAME(SetStatus)(io, "DELETE", 6) || (Fail() << "SetStatus(DELETE)", 0);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Close", 0);
  llvm::errs() << "end TestSequentialVariableUnformatted()\n";
}

void TestDirectFormatted() {
  llvm::errs() << "begin TestDirectFormatted()\n";
  // OPEN(NEWUNIT=unit,ACCESS='DIRECT',ACTION='READWRITE',&
  //   FORM='FORMATTED',RECL=8,STATUS='SCRATCH')
  auto io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(SetAccess)(io, "DIRECT", 6) || (Fail() << "SetAccess(DIRECT)", 0);
  IONAME(SetAction)
  (io, "READWRITE", 9) || (Fail() << "SetAction(READWRITE)", 0);
  IONAME(SetForm)
  (io, "FORMATTED", 9) || (Fail() << "SetForm(FORMATTED)", 0);
  static constexpr std::size_t recl{8};
  IONAME(SetRecl)(io, recl) || (Fail() << "SetRecl()", 0);
  IONAME(SetStatus)(io, "SCRATCH", 7) || (Fail() << "SetStatus(SCRATCH)", 0);
  int unit{-1};
  IONAME(GetNewUnit)(io, unit) || (Fail() << "GetNewUnit()", 0);
  llvm::errs() << "unit=" << unit << '\n';
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for OpenNewUnit", 0);
  static constexpr int records{10};
  static const char fmt[]{"(I4)"};
  for (int j{1}; j <= records; ++j) {
    // WRITE(UNIT=unit,FMT=fmt,REC=j) j
    io = IONAME(BeginExternalFormattedOutput)(
        fmt, sizeof fmt - 1, unit, __FILE__, __LINE__);
    IONAME(SetRec)(io, j) || (Fail() << "SetRec(" << j << ')', 0);
    IONAME(OutputInteger64)(io, j) || (Fail() << "OutputInteger64()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk || (Fail() << "EndIoStatement() for OutputInteger64", 0);
  }
  for (int j{records}; j >= 1; --j) {
    // READ(UNIT=unit,FMT=fmt,REC=j) n
    io = IONAME(BeginExternalFormattedInput)(
        fmt, sizeof fmt - 1, unit, __FILE__, __LINE__);
    IONAME(SetRec)(io, j) || (Fail() << "SetRec(" << j << ')', 0);
    std::int64_t buffer;
    IONAME(InputInteger)(io, buffer) || (Fail() << "InputInteger()", 0);
    IONAME(EndIoStatement)
    (io) == IostatOk || (Fail() << "EndIoStatement() for InputInteger", 0);
    if (buffer != j) {
      Fail() << "Read back " << buffer << " from direct formatted record " << j
             << ", expected " << j << '\n';
    }
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  IONAME(SetStatus)(io, "DELETE", 6) || (Fail() << "SetStatus(DELETE)", 0);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Close", 0);
  llvm::errs() << "end TestDirectformatted()\n";
}

void TestSequentialVariableFormatted() {
  llvm::errs() << "begin TestSequentialVariableFormatted()\n";
  // OPEN(NEWUNIT=unit,ACCESS='SEQUENTIAL',ACTION='READWRITE',&
  //   FORM='FORMATTED',STATUS='SCRATCH')
  auto io{IONAME(BeginOpenNewUnit)(__FILE__, __LINE__)};
  IONAME(SetAccess)
  (io, "SEQUENTIAL", 10) || (Fail() << "SetAccess(SEQUENTIAL)", 0);
  IONAME(SetAction)
  (io, "READWRITE", 9) || (Fail() << "SetAction(READWRITE)", 0);
  IONAME(SetForm)
  (io, "FORMATTED", 9) || (Fail() << "SetForm(FORMATTED)", 0);
  IONAME(SetStatus)(io, "SCRATCH", 7) || (Fail() << "SetStatus(SCRATCH)", 0);
  int unit{-1};
  IONAME(GetNewUnit)(io, unit) || (Fail() << "GetNewUnit()", 0);
  llvm::errs() << "unit=" << unit << '\n';
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for OpenNewUnit", 0);
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
      IONAME(OutputInteger64)
      (io, buffer[k]) || (Fail() << "OutputInteger64()", 0);
    }
    IONAME(EndIoStatement)
    (io) == IostatOk || (Fail() << "EndIoStatement() for OutputInteger64", 0);
  }
  // REWIND(UNIT=unit)
  io = IONAME(BeginRewind)(unit, __FILE__, __LINE__);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Rewind", 0);
  for (int j{1}; j <= records; ++j) {
    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // DO J=1,RECORDS; READ(UNIT=unit,FMT=fmt) n; check n; END DO
    io = IONAME(BeginExternalFormattedInput)(
        fmt, std::strlen(fmt), unit, __FILE__, __LINE__);
    std::int64_t check[records];
    for (int k{0}; k < j; ++k) {
      IONAME(InputInteger)(io, check[k]) || (Fail() << "InputInteger()", 0);
    }
    IONAME(EndIoStatement)
    (io) == IostatOk || (Fail() << "EndIoStatement() for InputInteger", 0);
    for (int k{0}; k < j; ++k) {
      if (buffer[k] != check[k]) {
        Fail() << "Read back [" << k << "]=" << check[k]
               << " from sequential variable formatted record " << j
               << ", expected " << buffer[k] << '\n';
      }
    }
  }
  for (int j{records}; j >= 1; --j) {
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for Backspace (before read)", 0);
    std::snprintf(fmt, sizeof fmt, "(%dI4)", j);
    // READ(UNIT=unit,FMT=fmt) n; check
    io = IONAME(BeginExternalFormattedInput)(
        fmt, std::strlen(fmt), unit, __FILE__, __LINE__);
    std::int64_t check[records];
    for (int k{0}; k < j; ++k) {
      IONAME(InputInteger)(io, check[k]) || (Fail() << "InputInteger()", 0);
    }
    IONAME(EndIoStatement)
    (io) == IostatOk || (Fail() << "EndIoStatement() for InputInteger", 0);
    for (int k{0}; k < j; ++k) {
      if (buffer[k] != check[k]) {
        Fail() << "Read back [" << k << "]=" << buffer[k]
               << " from sequential variable formatted record " << j
               << ", expected " << buffer[k] << '\n';
      }
    }
    // BACKSPACE(unit)
    io = IONAME(BeginBackspace)(unit, __FILE__, __LINE__);
    IONAME(EndIoStatement)
    (io) == IostatOk ||
        (Fail() << "EndIoStatement() for Backspace (after read)", 0);
  }
  // CLOSE(UNIT=unit,STATUS='DELETE')
  io = IONAME(BeginClose)(unit, __FILE__, __LINE__);
  IONAME(SetStatus)(io, "DELETE", 6) || (Fail() << "SetStatus(DELETE)", 0);
  IONAME(EndIoStatement)
  (io) == IostatOk || (Fail() << "EndIoStatement() for Close", 0);
  llvm::errs() << "end TestSequentialVariableFormatted()\n";
}

void TestStreamUnformatted() {
  // TODO
}

int main() {
  StartTests();
  TestDirectUnformatted();
  TestDirectUnformattedSwapped();
  TestSequentialFixedUnformatted();
  TestSequentialVariableUnformatted();
  TestDirectFormatted();
  TestSequentialVariableFormatted();
  TestStreamUnformatted();
  return EndTests();
}
