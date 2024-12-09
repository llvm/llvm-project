//===--- Builtins.h - Builtin function header -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines enum values for all the target-independent builtin
/// functions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_BUILTINS_H
#define LLVM_CLANG_BASIC_BUILTINS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cstring>

// VC++ defines 'alloca' as an object-like macro, which interferes with our
// builtins.
#undef alloca

namespace clang {
class TargetInfo;
class IdentifierTable;
class LangOptions;

enum LanguageID : uint16_t {
  GNU_LANG = 0x1,            // builtin requires GNU mode.
  C_LANG = 0x2,              // builtin for c only.
  CXX_LANG = 0x4,            // builtin for cplusplus only.
  OBJC_LANG = 0x8,           // builtin for objective-c and objective-c++
  MS_LANG = 0x10,            // builtin requires MS mode.
  OMP_LANG = 0x20,           // builtin requires OpenMP.
  CUDA_LANG = 0x40,          // builtin requires CUDA.
  COR_LANG = 0x80,           // builtin requires use of 'fcoroutine-ts' option.
  OCL_GAS = 0x100,           // builtin requires OpenCL generic address space.
  OCL_PIPE = 0x200,          // builtin requires OpenCL pipe.
  OCL_DSE = 0x400,           // builtin requires OpenCL device side enqueue.
  ALL_OCL_LANGUAGES = 0x800, // builtin for OCL languages.
  HLSL_LANG = 0x1000,        // builtin requires HLSL.
  ALL_LANGUAGES = C_LANG | CXX_LANG | OBJC_LANG, // builtin for all languages.
  ALL_GNU_LANGUAGES = ALL_LANGUAGES | GNU_LANG,  // builtin requires GNU mode.
  ALL_MS_LANGUAGES = ALL_LANGUAGES | MS_LANG     // builtin requires MS mode.
};

struct HeaderDesc {
  enum HeaderID : uint16_t {
#define HEADER(ID, NAME) ID,
#include "clang/Basic/BuiltinHeaders.def"
#undef HEADER
  } ID;

  constexpr HeaderDesc() : ID() {}
  constexpr HeaderDesc(HeaderID ID) : ID(ID) {}

  const char *getName() const;
};

namespace Builtin {
enum ID {
  NotBuiltin  = 0,      // This is not a builtin function.
#define BUILTIN(ID, TYPE, ATTRS) BI##ID,
#include "clang/Basic/Builtins.inc"
  FirstTSBuiltin
};

// The info used to represent each builtin.
struct Info {
  // Rather than store pointers to the string literals describing these four
  // aspects of builtins, we store offsets into a common string table.
  struct StrOffsets {
    int Name;
    int Type;
    int Attributes;
    int Features;
  } Offsets;

  HeaderDesc Header;
  LanguageID Langs;
};

// The storage for `N` builtins. This contains a single pointer to the string
// table used for these builtins and an array of metadata for each builtin.
template <size_t N> struct Storage {
  const char *StringTable;

  std::array<Info, N> Infos;

  // A constexpr function to construct the storage for a a given string table in
  // the first argument and an array in the second argument. This is *only*
  // expected to be used at compile time, we should mark it `consteval` when
  // available.
  //
  // The `Infos` array is particularly special. This function expects an array
  // of `Info` structs, where the string offsets of each entry refer to the
  // *sizes* of those strings rather than their offsets, and for the target
  // string to be in the provided string table at an offset the sum of all
  // previous string sizes. This function walks the `Infos` array computing the
  // running sum and replacing the sizes with the actual offsets in the string
  // table that should be used. This arrangement is designed to make it easy to
  // expand `.def` and `.inc` files with X-macros to construct both the string
  // table and the `Info` structs in the arguments to this function.
  static constexpr Storage<N> Make(const char *Strings,
                                   std::array<Info, N> Infos) {
    // Translate lengths to offsets.
    int Offset = 0;
    for (auto &I : Infos) {
      Info::StrOffsets NewOffsets = {};
      NewOffsets.Name = Offset;
      Offset += I.Offsets.Name;
      NewOffsets.Type = Offset;
      Offset += I.Offsets.Type;
      NewOffsets.Attributes = Offset;
      Offset += I.Offsets.Attributes;
      NewOffsets.Features = Offset;
      Offset += I.Offsets.Features;
      I.Offsets = NewOffsets;
    }
    return {Strings, Infos};
  }
};

// A detail macro used below to emit a string literal that, after string literal
// concatenation, ends up triggering the `-Woverlength-strings` warning. While
// the warning is useful in general to catch accidentally excessive strings,
// here we are creating them intentionally.
//
// This relies on a subtle aspect of `_Pragma`: that the *diagnostic* ones don't
// turn into actual tokens that would disrupt string literal concatenation.
#ifdef __clang__
#define CLANG_BUILTIN_DETAIL_STR_TABLE(S)                                      \
  _Pragma("clang diagnostic push")                                             \
      _Pragma("clang diagnostic ignored \"-Woverlength-strings\"")             \
          S _Pragma("clang diagnostic pop")
#else
#define CLANG_BUILTIN_DETAIL_STR_TABLE(S) S
#endif

// A macro that can be used with `Builtins.def` and similar files as an X-macro
// to add the string arguments to a builtin string table. This is typically the
// target for the `BUILTIN`, `LANGBUILTIN`, or `LIBBUILTIN` macros in those
// files.
#define CLANG_BUILTIN_STR_TABLE(ID, TYPE, ATTRS)                               \
  CLANG_BUILTIN_DETAIL_STR_TABLE(#ID "\0" TYPE "\0" ATTRS "\0" /*FEATURE*/ "\0")

// A macro that can be used with target builtin `.def` and `.inc` files as an
// X-macro to add the string arguments to a builtin string table. this is
// typically the target for the `TARGET_BUILTIN` macro.
#define CLANG_TARGET_BUILTIN_STR_TABLE(ID, TYPE, ATTRS, FEATURE)               \
  CLANG_BUILTIN_DETAIL_STR_TABLE(#ID "\0" TYPE "\0" ATTRS "\0" FEATURE "\0")

// A macro that can be used with target builtin `.def` and `.inc` files as an
// X-macro to add the string arguments to a builtin string table. this is
// typically the target for the `TARGET_HEADER_BUILTIN` macro. We can't delegate
// to `TARGET_BUILTIN` because the `FEATURE` string changes position.
#define CLANG_TARGET_HEADER_BUILTIN_STR_TABLE(ID, TYPE, ATTRS, HEADER, LANGS,  \
                                              FEATURE)                         \
  CLANG_BUILTIN_DETAIL_STR_TABLE(#ID "\0" TYPE "\0" ATTRS "\0" FEATURE "\0")

// A detail macro used internally to compute the desired string table
// `StrOffsets` struct for arguments to `Storage::Make`.
#define CLANG_BUILTIN_DETAIL_STR_OFFSETS(ID, TYPE, ATTRS)                      \
  Builtin::Info::StrOffsets {                                                  \
    sizeof(#ID), sizeof(TYPE), sizeof(ATTRS), sizeof("")                       \
  }

// A detail macro used internally to compute the desired string table
// `StrOffsets` struct for arguments to `Storage::Make`.
#define CLANG_TARGET_BUILTIN_DETAIL_STR_OFFSETS(ID, TYPE, ATTRS, FEATURE)      \
  Builtin::Info::StrOffsets {                                                  \
    sizeof(#ID), sizeof(TYPE), sizeof(ATTRS), sizeof(FEATURE)                  \
  }

// A set of macros that can be used with builtin `.def' files as an X-macro to
// create an `Info` struct for a particular builtin. It both computes the
// `StrOffsets` value for the string table (the lengths here, translated to
// offsets by the Storage::Make function), and the other metadata for each
// builtin.
//
// There is a corresponding macro for each of `BUILTIN`, `LANGBUILTIN`,
// `LIBBUILTIN`, `TARGET_BUILTIN`, and `TARGET_HEADER_BUILTIN`.
#define CLANG_BUILTIN_ENTRY(ID, TYPE, ATTRS)                                   \
  Builtin::Info{CLANG_BUILTIN_DETAIL_STR_OFFSETS(ID, TYPE, ATTRS),             \
                HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#define CLANG_LANGBUILTIN_ENTRY(ID, TYPE, ATTRS, LANG)                         \
  Builtin::Info{CLANG_BUILTIN_DETAIL_STR_OFFSETS(ID, TYPE, ATTRS),             \
                HeaderDesc::NO_HEADER, LANG},
#define CLANG_LIBBUILTIN_ENTRY(ID, TYPE, ATTRS, HEADER, LANG)                  \
  Builtin::Info{CLANG_BUILTIN_DETAIL_STR_OFFSETS(ID, TYPE, ATTRS),             \
                HeaderDesc::HEADER, LANG},
#define CLANG_TARGET_BUILTIN_ENTRY(ID, TYPE, ATTRS, FEATURE)                   \
  Builtin::Info{                                                               \
      CLANG_TARGET_BUILTIN_DETAIL_STR_OFFSETS(ID, TYPE, ATTRS, FEATURE),       \
      HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#define CLANG_TARGET_HEADER_BUILTIN_ENTRY(ID, TYPE, ATTRS, HEADER, LANG,       \
                                          FEATURE)                             \
  Builtin::Info{                                                               \
      CLANG_TARGET_BUILTIN_DETAIL_STR_OFFSETS(ID, TYPE, ATTRS, FEATURE),       \
      HeaderDesc::HEADER, LANG},

/// Holds information about both target-independent and
/// target-specific builtins, allowing easy queries by clients.
///
/// Builtins from an optional auxiliary target are stored in
/// AuxTSRecords. Their IDs are shifted up by TSRecords.size() and need to
/// be translated back with getAuxBuiltinID() before use.
class Context {
  const char *TSStrTable = nullptr;
  const char *AuxTSStrTable = nullptr;

  llvm::ArrayRef<Info> TSInfos;
  llvm::ArrayRef<Info> AuxTSInfos;

public:
  Context() = default;

  /// Perform target-specific initialization
  /// \param AuxTarget Target info to incorporate builtins from. May be nullptr.
  void InitializeTarget(const TargetInfo &Target, const TargetInfo *AuxTarget);

  /// Mark the identifiers for all the builtins with their
  /// appropriate builtin ID # and mark any non-portable builtin identifiers as
  /// such.
  void initializeBuiltins(IdentifierTable &Table, const LangOptions& LangOpts);

  /// Return the identifier name for the specified builtin,
  /// e.g. "__builtin_abs".
  llvm::StringRef getName(unsigned ID) const;

  /// Get the type descriptor string for the specified builtin.
  const char *getTypeString(unsigned ID) const;

  /// Get the attributes descriptor string for the specified builtin.
  const char *getAttributesString(unsigned ID) const;

  /// Return true if this function is a target-specific builtin.
  bool isTSBuiltin(unsigned ID) const {
    return ID >= Builtin::FirstTSBuiltin;
  }

  /// Return true if this function has no side effects.
  bool isPure(unsigned ID) const {
    return strchr(getAttributesString(ID), 'U') != nullptr;
  }

  /// Return true if this function has no side effects and doesn't
  /// read memory.
  bool isConst(unsigned ID) const {
    return strchr(getAttributesString(ID), 'c') != nullptr;
  }

  /// Return true if we know this builtin never throws an exception.
  bool isNoThrow(unsigned ID) const {
    return strchr(getAttributesString(ID), 'n') != nullptr;
  }

  /// Return true if we know this builtin never returns.
  bool isNoReturn(unsigned ID) const {
    return strchr(getAttributesString(ID), 'r') != nullptr;
  }

  /// Return true if we know this builtin can return twice.
  bool isReturnsTwice(unsigned ID) const {
    return strchr(getAttributesString(ID), 'j') != nullptr;
  }

  /// Returns true if this builtin does not perform the side-effects
  /// of its arguments.
  bool isUnevaluated(unsigned ID) const {
    return strchr(getAttributesString(ID), 'u') != nullptr;
  }

  /// Return true if this is a builtin for a libc/libm function,
  /// with a "__builtin_" prefix (e.g. __builtin_abs).
  bool isLibFunction(unsigned ID) const {
    return strchr(getAttributesString(ID), 'F') != nullptr;
  }

  /// Determines whether this builtin is a predefined libc/libm
  /// function, such as "malloc", where we know the signature a
  /// priori.
  /// In C, such functions behave as if they are predeclared,
  /// possibly with a warning on first use. In Objective-C and C++,
  /// they do not, but they are recognized as builtins once we see
  /// a declaration.
  bool isPredefinedLibFunction(unsigned ID) const {
    return strchr(getAttributesString(ID), 'f') != nullptr;
  }

  /// Returns true if this builtin requires appropriate header in other
  /// compilers. In Clang it will work even without including it, but we can emit
  /// a warning about missing header.
  bool isHeaderDependentFunction(unsigned ID) const {
    return strchr(getAttributesString(ID), 'h') != nullptr;
  }

  /// Determines whether this builtin is a predefined compiler-rt/libgcc
  /// function, such as "__clear_cache", where we know the signature a
  /// priori.
  bool isPredefinedRuntimeFunction(unsigned ID) const {
    return strchr(getAttributesString(ID), 'i') != nullptr;
  }

  /// Determines whether this builtin is a C++ standard library function
  /// that lives in (possibly-versioned) namespace std, possibly a template
  /// specialization, where the signature is determined by the standard library
  /// declaration.
  bool isInStdNamespace(unsigned ID) const {
    return strchr(getAttributesString(ID), 'z') != nullptr;
  }

  /// Determines whether this builtin can have its address taken with no
  /// special action required.
  bool isDirectlyAddressable(unsigned ID) const {
    // Most standard library functions can have their addresses taken. C++
    // standard library functions formally cannot in C++20 onwards, and when
    // we allow it, we need to ensure we instantiate a definition.
    return isPredefinedLibFunction(ID) && !isInStdNamespace(ID);
  }

  /// Determines whether this builtin has custom typechecking.
  bool hasCustomTypechecking(unsigned ID) const {
    return strchr(getAttributesString(ID), 't') != nullptr;
  }

  /// Determines whether a declaration of this builtin should be recognized
  /// even if the type doesn't match the specified signature.
  bool allowTypeMismatch(unsigned ID) const {
    return strchr(getAttributesString(ID), 'T') != nullptr ||
           hasCustomTypechecking(ID);
  }

  /// Determines whether this builtin has a result or any arguments which
  /// are pointer types.
  bool hasPtrArgsOrResult(unsigned ID) const {
    return strchr(getTypeString(ID), '*') != nullptr;
  }

  /// Return true if this builtin has a result or any arguments which are
  /// reference types.
  bool hasReferenceArgsOrResult(unsigned ID) const {
    return strchr(getTypeString(ID), '&') != nullptr ||
           strchr(getTypeString(ID), 'A') != nullptr;
  }

  /// If this is a library function that comes from a specific
  /// header, retrieve that header name.
  const char *getHeaderName(unsigned ID) const {
    return getInfo(ID).Header.getName();
  }

  /// Determine whether this builtin is like printf in its
  /// formatting rules and, if so, set the index to the format string
  /// argument and whether this function as a va_list argument.
  bool isPrintfLike(unsigned ID, unsigned &FormatIdx, bool &HasVAListArg);

  /// Determine whether this builtin is like scanf in its
  /// formatting rules and, if so, set the index to the format string
  /// argument and whether this function as a va_list argument.
  bool isScanfLike(unsigned ID, unsigned &FormatIdx, bool &HasVAListArg);

  /// Determine whether this builtin has callback behavior (see
  /// llvm::AbstractCallSites for details). If so, add the index to the
  /// callback callee argument and the callback payload arguments.
  bool performsCallback(unsigned ID,
                        llvm::SmallVectorImpl<int> &Encoding) const;

  /// Return true if this function has no side effects and doesn't
  /// read memory, except for possibly errno or raising FP exceptions.
  ///
  /// Such functions can be const when the MathErrno lang option and FP
  /// exceptions are disabled.
  bool isConstWithoutErrnoAndExceptions(unsigned ID) const {
    return strchr(getAttributesString(ID), 'e') != nullptr;
  }

  bool isConstWithoutExceptions(unsigned ID) const {
    return strchr(getAttributesString(ID), 'g') != nullptr;
  }

  const char *getRequiredFeatures(unsigned ID) const;

  unsigned getRequiredVectorWidth(unsigned ID) const;

  /// Return true if builtin ID belongs to AuxTarget.
  bool isAuxBuiltinID(unsigned ID) const {
    return ID >= (Builtin::FirstTSBuiltin + TSInfos.size());
  }

  /// Return real builtin ID (i.e. ID it would have during compilation
  /// for AuxTarget).
  unsigned getAuxBuiltinID(unsigned ID) const { return ID - TSInfos.size(); }

  /// Returns true if this is a libc/libm function without the '__builtin_'
  /// prefix.
  static bool isBuiltinFunc(llvm::StringRef Name);

  /// Returns true if this is a builtin that can be redeclared.  Returns true
  /// for non-builtins.
  bool canBeRedeclared(unsigned ID) const;

  /// Return true if this function can be constant evaluated by Clang frontend.
  bool isConstantEvaluated(unsigned ID) const {
    return strchr(getAttributesString(ID), 'E') != nullptr;
  }

  /// Returns true if this is an immediate (consteval) function
  bool isImmediate(unsigned ID) const {
    return strchr(getAttributesString(ID), 'G') != nullptr;
  }

private:
  std::pair<const char *, const Info &> getStrTableAndInfo(unsigned ID) const;

  const Info &getInfo(unsigned ID) const {
    return getStrTableAndInfo(ID).second;
  }

  /// Helper function for isPrintfLike and isScanfLike.
  bool isLike(unsigned ID, unsigned &FormatIdx, bool &HasVAListArg,
              const char *Fmt) const;
};

/// Returns true if the required target features of a builtin function are
/// enabled.
/// \p TargetFeatureMap maps a target feature to true if it is enabled and
///    false if it is disabled.
bool evaluateRequiredTargetFeatures(
    llvm::StringRef RequiredFatures,
    const llvm::StringMap<bool> &TargetFetureMap);

} // namespace Builtin

/// Kinds of BuiltinTemplateDecl.
enum BuiltinTemplateKind : int {
  /// This names the __make_integer_seq BuiltinTemplateDecl.
  BTK__make_integer_seq,

  /// This names the __type_pack_element BuiltinTemplateDecl.
  BTK__type_pack_element,

  /// This names the __builtin_common_type BuiltinTemplateDecl.
  BTK__builtin_common_type,
};

} // end namespace clang
#endif
