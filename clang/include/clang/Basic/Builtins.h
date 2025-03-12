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
#include "llvm/ADT/StringTable.h"
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
  NotBuiltin = 0, // This is not a builtin function.
#define GET_BUILTIN_ENUMERATORS
#include "clang/Basic/Builtins.inc"
#undef GET_BUILTIN_ENUMERATORS
  FirstTSBuiltin
};

struct InfosShard;

/// The info used to represent each builtin.
struct Info {
  // Rather than store pointers to the string literals describing these four
  // aspects of builtins, we store offsets into a common string table.
  struct StrOffsets {
    llvm::StringTable::Offset Name = {};
    llvm::StringTable::Offset Type = {};
    llvm::StringTable::Offset Attributes = {};

    // Defaults to the empty string offset.
    llvm::StringTable::Offset Features = {};
  } Offsets;

  HeaderDesc Header = HeaderDesc::NO_HEADER;
  LanguageID Langs = ALL_LANGUAGES;

  /// Get the name for the builtin represented by this `Info` object.
  ///
  /// Must be provided the `Shard` for this `Info` object.
  std::string getName(const InfosShard &Shard) const;
};

/// A constexpr function to construct an infos array from X-macros.
///
/// The input array uses the same data structure, but the offsets are actually
/// _lengths_ when input. This is all we can compute from the X-macro approach
/// to builtins. This function will convert these lengths into actual offsets to
/// a string table built up through sequentially appending strings with the
/// given lengths.
template <size_t N>
static constexpr std::array<Info, N> MakeInfos(std::array<Info, N> Infos) {
  // Translate lengths to offsets. We start past the initial empty string at
  // offset zero.
  unsigned Offset = 1;
  for (Info &I : Infos) {
    Info::StrOffsets NewOffsets = {};
    NewOffsets.Name = Offset;
    Offset += I.Offsets.Name.value();
    NewOffsets.Type = Offset;
    Offset += I.Offsets.Type.value();
    NewOffsets.Attributes = Offset;
    Offset += I.Offsets.Attributes.value();
    NewOffsets.Features = Offset;
    Offset += I.Offsets.Features.value();
    I.Offsets = NewOffsets;
  }
  return Infos;
}

/// A shard of a target's builtins string table and info.
///
/// Target builtins are sharded across multiple tables due to different
/// structures, origins, and also to improve the overall scaling by avoiding a
/// single table across all builtins.
struct InfosShard {
  const llvm::StringTable *Strings;
  llvm::ArrayRef<Info> Infos;

  llvm::StringLiteral NamePrefix = "";
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

// We require string tables to start with an empty string so that a `0` offset
// can always be used to refer to an empty string. To satisfy that when building
// string tables with X-macros, we use this start macro prior to expanding the
// X-macros.
#define CLANG_BUILTIN_STR_TABLE_START CLANG_BUILTIN_DETAIL_STR_TABLE("\0")

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
// `StrOffsets` struct for arguments to `MakeInfos`.
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
// offsets by the `MakeInfos` function), and the other metadata for each
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
  llvm::SmallVector<InfosShard> BuiltinShards;

  llvm::SmallVector<InfosShard> TargetShards;
  llvm::SmallVector<InfosShard> AuxTargetShards;

  unsigned NumTargetBuiltins = 0;
  unsigned NumAuxTargetBuiltins = 0;

public:
  Context();

  /// Perform target-specific initialization
  /// \param AuxTarget Target info to incorporate builtins from. May be nullptr.
  void InitializeTarget(const TargetInfo &Target, const TargetInfo *AuxTarget);

  /// Mark the identifiers for all the builtins with their
  /// appropriate builtin ID # and mark any non-portable builtin identifiers as
  /// such.
  void initializeBuiltins(IdentifierTable &Table, const LangOptions& LangOpts);

  /// Return the identifier name for the specified builtin,
  /// e.g. "__builtin_abs".
  std::string getName(unsigned ID) const;

  /// Return the identifier name for the specified builtin inside single quotes
  /// for a diagnostic, e.g. "'__builtin_abs'".
  std::string getQuotedName(unsigned ID) const;

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
    return ID >= (Builtin::FirstTSBuiltin + NumTargetBuiltins);
  }

  /// Return real builtin ID (i.e. ID it would have during compilation
  /// for AuxTarget).
  unsigned getAuxBuiltinID(unsigned ID) const { return ID - NumTargetBuiltins; }

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
  std::pair<const InfosShard &, const Info &>
  getShardAndInfo(unsigned ID) const;

  const Info &getInfo(unsigned ID) const { return getShardAndInfo(ID).second; }

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
