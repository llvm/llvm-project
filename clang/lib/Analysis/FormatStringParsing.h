#ifndef LLVM_CLANG_FORMAT_PARSING_H
#define LLVM_CLANG_FORMAT_PARSING_H

#include "clang/Analysis/Analyses/FormatString.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

class LangOptions;

template <typename T>
class UpdateOnReturn {
  T &ValueToUpdate;
  const T &ValueToCopy;
public:
  UpdateOnReturn(T &valueToUpdate, const T &valueToCopy)
    : ValueToUpdate(valueToUpdate), ValueToCopy(valueToCopy) {}

  ~UpdateOnReturn() {
    ValueToUpdate = ValueToCopy;
  }
};

namespace analyze_format_string {
  
OptionalAmount ParseAmount(const char *&Beg, const char *E);
OptionalAmount ParseNonPositionAmount(const char *&Beg, const char *E,
                                      unsigned &argIndex);

OptionalAmount ParsePositionAmount(FormatStringHandler &H,
                                   const char *Start, const char *&Beg,
                                   const char *E, PositionContext p);
  
bool ParseFieldWidth(FormatStringHandler &H,
                     FormatSpecifier &CS,
                     const char *Start, const char *&Beg, const char *E,
                     unsigned *argIndex);
    
bool ParseArgPosition(FormatStringHandler &H,
                      FormatSpecifier &CS, const char *Start,
                      const char *&Beg, const char *E);

/// Returns true if a LengthModifier was parsed and installed in the
/// FormatSpecifier& argument, and false otherwise.
bool ParseLengthModifier(FormatSpecifier &FS, const char *&Beg, const char *E,
                         const LangOptions &LO, bool IsScanf = false);
  
template <typename T> class SpecifierResult {
  T FS;
  const char *Start;
  bool Stop;
public:
  SpecifierResult(bool stop = false)
  : Start(0), Stop(stop) {}
  SpecifierResult(const char *start,
                  const T &fs)
  : FS(fs), Start(start), Stop(false) {}
  
  const char *getStart() const { return Start; }
  bool shouldStop() const { return Stop; }
  bool hasValue() const { return Start != 0; }
  const T &getValue() const {
    assert(hasValue());
    return FS;
  }
  const T &getValue() { return FS; }
};
  
} // end analyze_format_string namespace
} // end clang namespace

#endif
