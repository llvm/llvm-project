//===-- xray_fdr_log_printer_tool.cc --------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a function call tracing system.
//
//===----------------------------------------------------------------------===//

#include "xray_fdr_logging.h"
#include "xray_fdr_logging_impl.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "xray/xray_records.h"

// Writes out xray fdr mode log records to stdout based on a sequence of
// formatted data read from stdin.
//
// Interprets an adhoc format of Top Level Types and parameter maps of the form:
//
// RecordType : { Parameter1 = Value, Parameter2 = value , Parameter3 = value}
// OtherRecordType : { ParameterOne = Value }
//
// Each line corresponds to a type of record written by the Xray Flight Data
// Recorder mode to a buffer. This program synthesizes records in the FDR binary
// format and writes them to std::cout.

namespace {

/// A crude lexer to read tokens and skip over whitespace.
class TokenReader {
public:
  TokenReader() : LastDelimPresent(false), FoundEof(false), LastDelim(0) {}
  std::string readToken(std::istream &Stream);
  bool hasLastDelim() const { return LastDelimPresent; }
  char getLastDelim() const { return LastDelim; }
  void setLastDelim(char Delim) {
    LastDelimPresent = true;
    LastDelim = Delim;
  }
  void clearLastDelim() {
    LastDelimPresent = false;
    LastDelim = 0;
  }
  bool isEof() { return FoundEof; }
  void setFoundEof(bool eof) { FoundEof = eof; }

private:
  bool LastDelimPresent;
  bool FoundEof;
  char LastDelim;
};

// Globally tracks whether we reached EOF and caches delimiters that qualify as
// tokens.
static TokenReader TokenReader{};

bool isWhitespace(char c) {
  // Hardcode the whitespace characters we will not return as tokens even though
  // they are token delimiters.
  static const std::vector<char> Whitespace{' ', '\n', '\t'};
  return std::find(Whitespace.begin(), Whitespace.end(), c) != Whitespace.end();
}

bool isDelimiter(char c) {
  // Hardcode a set of token delimiters.
  static const std::vector<char> Delimiters{' ',  ':', ',', '\n',
                                            '\t', '{', '}', '='};
  return std::find(Delimiters.begin(), Delimiters.end(), c) != Delimiters.end();
}

std::string TokenReader::readToken(std::istream &Stream) {
  // If on the last call we read a trailing delimiter that also qualifies as a
  // token, return it now.
  if (hasLastDelim()) {
    char Token = getLastDelim();
    clearLastDelim();
    return std::string{Token};
  }

  std::stringstream Builder{};
  char c;
  c = Stream.get();
  while (!isDelimiter(c) && !Stream.eof()) {
    Builder << c;
    c = Stream.get();
  }

  setFoundEof(Stream.eof());

  std::string Token = Builder.str();

  if (Token.empty()) {
    // We read a whitespace delimiter only. Skip over it.
    if (!isEof() && isWhitespace(c)) {
      return readToken(Stream);
    } else if (isWhitespace(c)) {
      // We only read a whitespace delimiter.
      return "";
    } else {
      // We read a delimiter that matters as a token.
      return std::string{c};
    }
  }

  // If we found a delimiter that's a valid token. Store it to return as the
  // next token.
  if (!isWhitespace(c))
    setLastDelim(c);

  return Token;
}

// Reads an expected token or dies a gruesome death.
void eatExpectedToken(std::istream &Stream, const std::string &Expected) {
  std::string Token = TokenReader.readToken(Stream);
  if (Token.compare(Expected) != 0) {
    std::cerr << "Expecting token '" << Expected << "'. Found '" << Token
              << "'.\n";
    std::exit(1);
  }
}

// Constructs a map of key value pairs from a token stream.
// Expects to read an expression of the form:
//
// { a = b, c = d, e = f}
//
// If not, the driver will crash.
std::map<std::string, std::string> readMap(std::istream &Stream) {
  using StrMap = std::map<std::string, std::string>;
  using StrVector = std::vector<std::string>;

  eatExpectedToken(Stream, "{");
  StrVector TokenList{};

  while (!TokenReader.isEof()) {
    std::string CurrentToken = TokenReader.readToken(Stream);
    if (CurrentToken.compare("}") == 0) {
      break;
    }
    TokenList.push_back(CurrentToken);
    if (TokenReader.isEof()) {
      std::cerr << "Got EOF while building a param map.\n";
      std::exit(1);
    }
  }

  if (TokenList.size() == 0) {
    StrMap EmptyMap{};
    return EmptyMap;
  }
  if (TokenList.size() % 4 != 3) {
    std::cerr << "Error while building token map. Expected triples of tokens "
                 "in the form 'a = b' separated by commas.\n";
    std::exit(1);
  }

  StrMap TokenMap{};
  std::size_t ElementIndex = 0;
  for (; ElementIndex < TokenList.size(); ElementIndex += 4) {
    if (TokenList[ElementIndex + 1].compare("=") != 0) {
      std::cerr << "Expected an assignment when building a param map.\n";
      std::exit(1);
    }
    TokenMap[TokenList[ElementIndex]] = TokenList[ElementIndex + 2];
    if (ElementIndex + 3 < TokenList.size()) {
      if (TokenList[ElementIndex + 3].compare(",") != 0) {
        std::cerr << "Expected assignment statements to be separated by commas."
                  << "\n";
        std::exit(1);
      }
    }
  }
  return TokenMap;
}

std::string getOrDie(const std::map<std::string, std::string> &Lookup,
                     const std::string &Key) {
  auto MapIter = Lookup.find(Key);
  if (MapIter == Lookup.end()) {
    std::cerr << "Expected key '" << Key << "'. Was not found.\n";
    std::exit(1);
  }
  return MapIter->second;
}

// Reads a numeric type from a string token through the magic of
// std::stringstream.
template <typename NT> struct NumberParser {
  static NT parse(const std::string &Input) {
    NT Number = 0;
    std::stringstream Stream(Input);
    Stream >> Number;
    return Number;
  }
};

void writeNewBufferOrDie(std::istream &Input) {
  auto TokenMap = readMap(Input);
  pid_t Tid = NumberParser<pid_t>::parse(getOrDie(TokenMap, "Tid"));
  time_t Time = NumberParser<time_t>::parse(getOrDie(TokenMap, "time"));
  timespec TimeSpec = {Time, 0};
  constexpr const size_t OutputSize = 32;
  std::array<char, OutputSize> Buffer{};
  char *MemPtr = Buffer.data();
  __xray::__xray_fdr_internal::writeNewBufferPreamble(Tid, TimeSpec, MemPtr);
  std::cout.write(Buffer.data(), OutputSize);
}

void writeNewCPUIdOrDie(std::istream &Input) {
  auto TokenMap = readMap(Input);
  uint16_t CPU = NumberParser<uint16_t>::parse(getOrDie(TokenMap, "CPU"));
  uint64_t TSC = NumberParser<uint64_t>::parse(getOrDie(TokenMap, "TSC"));
  constexpr const size_t OutputSize = 16;
  std::array<char, OutputSize> Buffer{};
  char *MemPtr = Buffer.data();
  __xray::__xray_fdr_internal::writeNewCPUIdMetadata(CPU, TSC, MemPtr);
  std::cout.write(Buffer.data(), OutputSize);
}

void writeEOBOrDie(std::istream &Input) {
  auto TokenMap = readMap(Input);
  constexpr const size_t OutputSize = 16;
  std::array<char, OutputSize> Buffer{};
  char *MemPtr = Buffer.data();
  __xray::__xray_fdr_internal::writeEOBMetadata(MemPtr);
  std::cout.write(Buffer.data(), OutputSize);
}

void writeTSCWrapOrDie(std::istream &Input) {
  auto TokenMap = readMap(Input);
  uint64_t TSC = NumberParser<uint64_t>::parse(getOrDie(TokenMap, "TSC"));
  constexpr const size_t OutputSize = 16;
  std::array<char, OutputSize> Buffer{};
  char *MemPtr = Buffer.data();
  __xray::__xray_fdr_internal::writeTSCWrapMetadata(TSC, MemPtr);
  std::cout.write(Buffer.data(), OutputSize);
}

XRayEntryType decodeEntryType(const std::string &EntryTypeStr) {
  if (EntryTypeStr.compare("Entry") == 0) {
    return XRayEntryType::ENTRY;
  } else if (EntryTypeStr.compare("LogArgsEntry") == 0) {
    return XRayEntryType::LOG_ARGS_ENTRY;
  } else if (EntryTypeStr.compare("Exit") == 0) {
    return XRayEntryType::EXIT;
  } else if (EntryTypeStr.compare("Tail") == 0) {
    return XRayEntryType::TAIL;
  }
  std::cerr << "Illegal entry type " << EntryTypeStr << ".\n";
  std::exit(1);
}

void writeFunctionOrDie(std::istream &Input) {
  auto TokenMap = readMap(std::cin);
  int FuncId = NumberParser<int>::parse(getOrDie(TokenMap, "FuncId"));
  uint32_t TSCDelta =
      NumberParser<uint32_t>::parse(getOrDie(TokenMap, "TSCDelta"));
  std::string EntryType = getOrDie(TokenMap, "EntryType");
  XRayEntryType XrayEntryType = decodeEntryType(EntryType);
  constexpr const size_t OutputSize = 8;
  std::array<char, OutputSize> Buffer{};
  char *MemPtr = Buffer.data();
  __xray::__xray_fdr_internal::writeFunctionRecord(FuncId, TSCDelta,
                                                   XrayEntryType, MemPtr);
  std::cout.write(Buffer.data(), OutputSize);
}

} // namespace

int main(int argc, char **argv) {
  std::map<std::string, std::function<void(std::istream &)>> TopLevelRecordMap;
  TopLevelRecordMap["NewBuffer"] = writeNewBufferOrDie;
  TopLevelRecordMap["NewCPU"] = writeNewCPUIdOrDie;
  TopLevelRecordMap["EOB"] = writeEOBOrDie;
  TopLevelRecordMap["TSCWrap"] = writeTSCWrapOrDie;
  TopLevelRecordMap["Function"] = writeFunctionOrDie;

  // Write file header
  //
  //   (2)   uint16 : version
  //   (2)   uint16 : type
  //   (4)   uint32 : bitfield
  //   (8)   uint64 : cycle frequency
  //   (16)  -      : padding
  uint16_t HeaderVersion = 1;
  uint16_t HeaderType = 1;
  uint32_t Bitfield = 3;
  uint64_t CycleFreq = 42;
  constexpr const size_t HeaderSize = 32;
  std::array<char, HeaderSize> Header{};
  std::memcpy(Header.data(), &HeaderVersion, sizeof(HeaderVersion));
  std::memcpy(Header.data() + 2, &HeaderType, sizeof(HeaderType));
  std::memcpy(Header.data() + 4, &Bitfield, sizeof(Bitfield));
  std::memcpy(Header.data() + 8, &CycleFreq, sizeof(CycleFreq));
  std::cout.write(Header.data(), HeaderSize);

  std::string CurrentToken;
  while (true) {
    CurrentToken = TokenReader.readToken(std::cin);
    if (TokenReader.isEof())
      break;
    auto MapIter = TopLevelRecordMap.find(CurrentToken);
    if (MapIter != TopLevelRecordMap.end()) {
      eatExpectedToken(std::cin, ":");
      if (TokenReader.isEof()) {
        std::cerr << "Got eof when expecting to read a map.\n";
        std::exit(1);
      }
      MapIter->second(std::cin);
    } else {
      std::cerr << "Got bad top level instruction '" << CurrentToken << "'.\n";
      std::exit(1);
    }
  }
  return 0;
}
