//===--- SanitizerArgs.h - Arguments for sanitizer tools  -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_LIB_DRIVER_SANITIZERARGS_H_
#define CLANG_LIB_DRIVER_SANITIZERARGS_H_

#include "clang/Driver/ArgList.h"

namespace clang {
namespace driver {

class SanitizerArgs {
  /// Assign ordinals to sanitizer flags. We'll use the ordinal values as
  /// bit positions within \c Kind.
  enum SanitizeOrdinal {
#define SANITIZER(NAME, ID) SO_##ID,
#include "clang/Basic/Sanitizers.def"
    SO_Count
  };

  /// Bugs to catch at runtime.
  enum SanitizeKind {
#define SANITIZER(NAME, ID) ID = 1 << SO_##ID,
#define SANITIZER_GROUP(NAME, ID, ALIAS) ID = ALIAS,
#include "clang/Basic/Sanitizers.def"
    NeedsAsanRt = Address,
    NeedsTsanRt = Thread,
    NeedsUbsanRt = Undefined
  };
  unsigned Kind;

 public:
  SanitizerArgs() : Kind(0) {}
  /// Parses the sanitizer arguments from an argument list.
  SanitizerArgs(const Driver &D, const ArgList &Args);

  bool needsAsanRt() const { return Kind & NeedsAsanRt; }
  bool needsTsanRt() const { return Kind & NeedsTsanRt; }
  bool needsUbsanRt() const { return Kind & NeedsUbsanRt; }

  bool sanitizesVptr() const { return Kind & Vptr; }
  
  void addArgs(const ArgList &Args, ArgStringList &CmdArgs) const {
    if (!Kind)
      return;
    llvm::SmallString<256> SanitizeOpt("-fsanitize=");
#define SANITIZER(NAME, ID) \
    if (Kind & ID) \
      SanitizeOpt += NAME ",";
#include "clang/Basic/Sanitizers.def"
    SanitizeOpt.pop_back();
    CmdArgs.push_back(Args.MakeArgString(SanitizeOpt));
  }

 private:
  /// Parse a single value from a -fsanitize= or -fno-sanitize= value list.
  /// Returns a member of the \c SanitizeKind enumeration, or \c 0 if \p Value
  /// is not known.
  static unsigned parse(const char *Value) {
    return llvm::StringSwitch<SanitizeKind>(Value)
#define SANITIZER(NAME, ID) .Case(NAME, ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS) .Case(NAME, ID)
#include "clang/Basic/Sanitizers.def"
      .Default(SanitizeKind());
  }

  /// Parse a -fsanitize= or -fno-sanitize= argument's values, diagnosing any
  /// invalid components.
  static unsigned parse(const Driver &D, const Arg *A) {
    unsigned Kind = 0;
    for (unsigned I = 0, N = A->getNumValues(); I != N; ++I) {
      if (unsigned K = parse(A->getValue(I)))
        Kind |= K;
      else
        D.Diag(diag::err_drv_unsupported_option_argument)
          << A->getOption().getName() << A->getValue(I);
    }
    return Kind;
  }

  /// Produce an argument string from argument \p A, which shows how it provides
  /// a value in \p Mask. For instance, the argument
  /// "-fsanitize=address,alignment" with mask \c NeedsUbsanRt would produce
  /// "-fsanitize=alignment".
  static std::string describeSanitizeArg(const ArgList &Args, const Arg *A,
                                         unsigned Mask) {
    if (!A->getOption().matches(options::OPT_fsanitize_EQ))
      return A->getAsString(Args);

    for (unsigned I = 0, N = A->getNumValues(); I != N; ++I)
      if (parse(A->getValue(I)) & Mask)
        return std::string("-fsanitize=") + A->getValue(I);

    llvm_unreachable("arg didn't provide expected value");
  }
};

}  // namespace driver
}  // namespace clang

#endif // CLANG_LIB_DRIVER_SANITIZERARGS_H_
