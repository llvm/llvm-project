//===--- FormatStringConverter.cpp - clang-tidy----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the FormatStringConverter class which is used to convert
/// printf format strings to C++ std::formatter format strings.
///
//===----------------------------------------------------------------------===//

#include "FormatStringConverter.h"
#include "../utils/FixItHintUtils.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Lex/Lexer.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace clang::ast_matchers;
using namespace clang::analyze_printf;

namespace clang::tidy::utils {
using clang::analyze_format_string::ConversionSpecifier;

/// Is the passed type the actual "char" type, whether that be signed or
/// unsigned, rather than explicit signed char or unsigned char types.
static bool isRealCharType(const clang::QualType &Ty) {
  using namespace clang;
  const Type *DesugaredType = Ty->getUnqualifiedDesugaredType();
  if (const auto *BT = llvm::dyn_cast<BuiltinType>(DesugaredType))
    return (BT->getKind() == BuiltinType::Char_U ||
            BT->getKind() == BuiltinType::Char_S);
  return false;
}

/// If possible, return the text name of the signed type that corresponds to the
/// passed integer type. If the passed type is already signed then its name is
/// just returned. Only supports BuiltinTypes.
static std::optional<std::string>
getCorrespondingSignedTypeName(const clang::QualType &QT) {
  using namespace clang;
  const auto UQT = QT.getUnqualifiedType();
  if (const auto *BT = llvm::dyn_cast<BuiltinType>(UQT)) {
    switch (BT->getKind()) {
    case BuiltinType::UChar:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::Char_S:
      return "signed char";
    case BuiltinType::UShort:
    case BuiltinType::Short:
      return "short";
    case BuiltinType::UInt:
    case BuiltinType::Int:
      return "int";
    case BuiltinType::ULong:
    case BuiltinType::Long:
      return "long";
    case BuiltinType::ULongLong:
    case BuiltinType::LongLong:
      return "long long";
    default:
      llvm::dbgs() << "Unknown corresponding signed type for BuiltinType '"
                   << QT.getAsString() << "'\n";
      return std::nullopt;
    }
  }

  // Deal with fixed-width integer types from <cstdint>. Use std:: prefix only
  // if the argument type does.
  const std::string TypeName = UQT.getAsString();
  StringRef SimplifiedTypeName{TypeName};
  const bool InStd = SimplifiedTypeName.consume_front("std::");
  const StringRef Prefix = InStd ? "std::" : "";

  if (SimplifiedTypeName.starts_with("uint") &&
      SimplifiedTypeName.ends_with("_t"))
    return (Twine(Prefix) + SimplifiedTypeName.drop_front()).str();

  if (SimplifiedTypeName == "size_t")
    return (Twine(Prefix) + "ssize_t").str();

  llvm::dbgs() << "Unknown corresponding signed type for non-BuiltinType '"
               << UQT.getAsString() << "'\n";
  return std::nullopt;
}

/// If possible, return the text name of the unsigned type that corresponds to
/// the passed integer type. If the passed type is already unsigned then its
/// name is just returned. Only supports BuiltinTypes.
static std::optional<std::string>
getCorrespondingUnsignedTypeName(const clang::QualType &QT) {
  using namespace clang;
  const auto UQT = QT.getUnqualifiedType();
  if (const auto *BT = llvm::dyn_cast<BuiltinType>(UQT)) {
    switch (BT->getKind()) {
    case BuiltinType::SChar:
    case BuiltinType::Char_S:
    case BuiltinType::UChar:
    case BuiltinType::Char_U:
      return "unsigned char";
    case BuiltinType::Short:
    case BuiltinType::UShort:
      return "unsigned short";
    case BuiltinType::Int:
    case BuiltinType::UInt:
      return "unsigned int";
    case BuiltinType::Long:
    case BuiltinType::ULong:
      return "unsigned long";
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
      return "unsigned long long";
    default:
      llvm::dbgs() << "Unknown corresponding unsigned type for BuiltinType '"
                   << UQT.getAsString() << "'\n";
      return std::nullopt;
    }
  }

  // Deal with fixed-width integer types from <cstdint>. Use std:: prefix only
  // if the argument type does.
  const std::string TypeName = UQT.getAsString();
  StringRef SimplifiedTypeName{TypeName};
  const bool InStd = SimplifiedTypeName.consume_front("std::");
  const StringRef Prefix = InStd ? "std::" : "";

  if (SimplifiedTypeName.starts_with("int") &&
      SimplifiedTypeName.ends_with("_t"))
    return (Twine(Prefix) + "u" + SimplifiedTypeName).str();

  if (SimplifiedTypeName == "ssize_t")
    return (Twine(Prefix) + "size_t").str();
  if (SimplifiedTypeName == "ptrdiff_t")
    return (Twine(Prefix) + "size_t").str();

  llvm::dbgs() << "Unknown corresponding unsigned type for non-BuiltinType '"
               << UQT.getAsString() << "'\n";
  return std::nullopt;
}

static std::optional<std::string>
castTypeForArgument(ConversionSpecifier::Kind ArgKind,
                    const clang::QualType &QT) {
  if (ArgKind == ConversionSpecifier::Kind::uArg)
    return getCorrespondingUnsignedTypeName(QT);
  return getCorrespondingSignedTypeName(QT);
}

static bool isMatchingSignedness(ConversionSpecifier::Kind ArgKind,
                                 const clang::QualType &ArgType) {
  if (const auto *BT = llvm::dyn_cast<BuiltinType>(ArgType)) {
    // Unadorned char never matches any expected signedness since it
    // could be signed or unsigned.
    const auto ArgTypeKind = BT->getKind();
    if (ArgTypeKind == BuiltinType::Char_U ||
        ArgTypeKind == BuiltinType::Char_S)
      return false;
  }

  if (ArgKind == ConversionSpecifier::Kind::uArg)
    return ArgType->isUnsignedIntegerType();
  return ArgType->isSignedIntegerType();
}

namespace {
AST_MATCHER(clang::QualType, isRealChar) {
  return clang::tidy::utils::isRealCharType(Node);
}
} // namespace

static bool castMismatchedIntegerTypes(const CallExpr *Call, bool StrictMode) {
  /// For printf-style functions, the signedness of the type printed is
  /// indicated by the corresponding type in the format string.
  /// std::print will determine the signedness from the type of the
  /// argument. This means that it is necessary to generate a cast in
  /// StrictMode to ensure that the exact behaviour is maintained.
  /// However, for templated functions like absl::PrintF and
  /// fmt::printf, the signedness of the type printed is also taken from
  /// the actual argument like std::print, so such casts are never
  /// necessary. printf-style functions are variadic, whereas templated
  /// ones aren't, so we can use that to distinguish between the two
  /// cases.
  if (StrictMode) {
    const FunctionDecl *FuncDecl = Call->getDirectCallee();
    assert(FuncDecl);
    return FuncDecl->isVariadic();
  }
  return false;
}

FormatStringConverter::FormatStringConverter(ASTContext *ContextIn,
                                             const CallExpr *Call,
                                             unsigned FormatArgOffset,
                                             const Configuration ConfigIn,
                                             const LangOptions &LO)
    : Context(ContextIn), Config(ConfigIn),
      CastMismatchedIntegerTypes(
          castMismatchedIntegerTypes(Call, ConfigIn.StrictMode)),
      Args(Call->getArgs()), NumArgs(Call->getNumArgs()),
      ArgsOffset(FormatArgOffset + 1), LangOpts(LO) {
  assert(ArgsOffset <= NumArgs);
  FormatExpr = llvm::dyn_cast<StringLiteral>(
      Args[FormatArgOffset]->IgnoreImplicitAsWritten());
  if (!FormatExpr || !FormatExpr->isOrdinary()) {
    // Function must have a narrow string literal as its first argument.
    conversionNotPossible("first argument is not a narrow string literal");
    return;
  }
  PrintfFormatString = FormatExpr->getString();

  // Assume that the output will be approximately the same size as the input,
  // but perhaps with a few escapes expanded.
  const size_t EstimatedGrowth = 8;
  StandardFormatString.reserve(PrintfFormatString.size() + EstimatedGrowth);
  StandardFormatString.push_back('\"');

  const bool IsFreeBsdkPrintf = false;

  using clang::analyze_format_string::ParsePrintfString;
  ParsePrintfString(*this, PrintfFormatString.data(),
                    PrintfFormatString.data() + PrintfFormatString.size(),
                    LangOpts, Context->getTargetInfo(), IsFreeBsdkPrintf);
  finalizeFormatText();
}

void FormatStringConverter::emitAlignment(const PrintfSpecifier &FS,
                                          std::string &FormatSpec) {
  ConversionSpecifier::Kind ArgKind = FS.getConversionSpecifier().getKind();

  // We only care about alignment if a field width is specified
  if (FS.getFieldWidth().getHowSpecified() != OptionalAmount::NotSpecified) {
    if (ArgKind == ConversionSpecifier::sArg) {
      // Strings are left-aligned by default with std::format, so we only
      // need to emit an alignment if this one needs to be right aligned.
      if (!FS.isLeftJustified())
        FormatSpec.push_back('>');
    } else {
      // Numbers are right-aligned by default with std::format, so we only
      // need to emit an alignment if this one needs to be left aligned.
      if (FS.isLeftJustified())
        FormatSpec.push_back('<');
    }
  }
}

void FormatStringConverter::emitSign(const PrintfSpecifier &FS,
                                     std::string &FormatSpec) {
  const ConversionSpecifier Spec = FS.getConversionSpecifier();

  // Ignore on something that isn't numeric. For printf it's would be a
  // compile-time warning but ignored at runtime, but for std::format it
  // ought to be a compile-time error.
  if (Spec.isAnyIntArg() || Spec.isDoubleArg()) {
    // + is preferred to ' '
    if (FS.hasPlusPrefix())
      FormatSpec.push_back('+');
    else if (FS.hasSpacePrefix())
      FormatSpec.push_back(' ');
  }
}

void FormatStringConverter::emitAlternativeForm(const PrintfSpecifier &FS,
                                                std::string &FormatSpec) {
  if (FS.hasAlternativeForm()) {
    switch (FS.getConversionSpecifier().getKind()) {
    case ConversionSpecifier::Kind::aArg:
    case ConversionSpecifier::Kind::AArg:
    case ConversionSpecifier::Kind::eArg:
    case ConversionSpecifier::Kind::EArg:
    case ConversionSpecifier::Kind::fArg:
    case ConversionSpecifier::Kind::FArg:
    case ConversionSpecifier::Kind::gArg:
    case ConversionSpecifier::Kind::GArg:
    case ConversionSpecifier::Kind::xArg:
    case ConversionSpecifier::Kind::XArg:
    case ConversionSpecifier::Kind::oArg:
      FormatSpec.push_back('#');
      break;
    default:
      // Alternative forms don't exist for other argument kinds
      break;
    }
  }
}

void FormatStringConverter::emitFieldWidth(const PrintfSpecifier &FS,
                                           std::string &FormatSpec) {
  {
    const OptionalAmount FieldWidth = FS.getFieldWidth();
    switch (FieldWidth.getHowSpecified()) {
    case OptionalAmount::NotSpecified:
      break;
    case OptionalAmount::Constant:
      FormatSpec.append(llvm::utostr(FieldWidth.getConstantAmount()));
      break;
    case OptionalAmount::Arg:
      FormatSpec.push_back('{');
      if (FieldWidth.usesPositionalArg()) {
        // std::format argument identifiers are zero-based, whereas printf
        // ones are one based.
        assert(FieldWidth.getPositionalArgIndex() > 0U);
        FormatSpec.append(llvm::utostr(FieldWidth.getPositionalArgIndex() - 1));
      }
      FormatSpec.push_back('}');
      break;
    case OptionalAmount::Invalid:
      break;
    }
  }
}

void FormatStringConverter::emitPrecision(const PrintfSpecifier &FS,
                                          std::string &FormatSpec) {
  const OptionalAmount FieldPrecision = FS.getPrecision();
  switch (FieldPrecision.getHowSpecified()) {
  case OptionalAmount::NotSpecified:
    break;
  case OptionalAmount::Constant:
    FormatSpec.push_back('.');
    FormatSpec.append(llvm::utostr(FieldPrecision.getConstantAmount()));
    break;
  case OptionalAmount::Arg:
    FormatSpec.push_back('.');
    FormatSpec.push_back('{');
    if (FieldPrecision.usesPositionalArg()) {
      // std::format argument identifiers are zero-based, whereas printf
      // ones are one based.
      assert(FieldPrecision.getPositionalArgIndex() > 0U);
      FormatSpec.append(
          llvm::utostr(FieldPrecision.getPositionalArgIndex() - 1));
    }
    FormatSpec.push_back('}');
    break;
  case OptionalAmount::Invalid:
    break;
  }
}

void FormatStringConverter::maybeRotateArguments(const PrintfSpecifier &FS) {
  unsigned ArgCount = 0;
  const OptionalAmount FieldWidth = FS.getFieldWidth();
  const OptionalAmount FieldPrecision = FS.getPrecision();

  if (FieldWidth.getHowSpecified() == OptionalAmount::Arg &&
      !FieldWidth.usesPositionalArg())
    ++ArgCount;
  if (FieldPrecision.getHowSpecified() == OptionalAmount::Arg &&
      !FieldPrecision.usesPositionalArg())
    ++ArgCount;

  if (ArgCount)
    ArgRotates.emplace_back(FS.getArgIndex() + ArgsOffset, ArgCount);
}

void FormatStringConverter::emitStringArgument(unsigned ArgIndex,
                                               const Expr *Arg) {
  // If the argument is the result of a call to std::string::c_str() or
  // data() with a return type of char then we can remove that call and
  // pass the std::string directly. We don't want to do so if the return
  // type is not a char pointer (though it's unlikely that such code would
  // compile without warnings anyway.) See RedundantStringCStrCheck.

  if (!StringCStrCallExprMatcher) {
    // Lazily create the matcher
    const auto StringDecl = type(hasUnqualifiedDesugaredType(recordType(
        hasDeclaration(cxxRecordDecl(hasName("::std::basic_string"))))));
    const auto StringExpr = expr(
        anyOf(hasType(StringDecl), hasType(qualType(pointsTo(StringDecl)))));

    StringCStrCallExprMatcher =
        cxxMemberCallExpr(
            on(StringExpr.bind("arg")), callee(memberExpr().bind("member")),
            callee(cxxMethodDecl(hasAnyName("c_str", "data"),
                                 returns(pointerType(pointee(isRealChar()))))))
            .bind("call");
  }

  auto CStrMatches = match(*StringCStrCallExprMatcher, *Arg, *Context);
  if (CStrMatches.size() == 1)
    ArgCStrRemovals.push_back(CStrMatches.front());
  else if (Arg->getType()->isPointerType()) {
    const QualType Pointee = Arg->getType()->getPointeeType();
    // printf is happy to print signed char and unsigned char strings, but
    // std::format only likes char strings.
    if (Pointee->isCharType() && !isRealCharType(Pointee))
      ArgFixes.emplace_back(ArgIndex, "reinterpret_cast<const char *>(");
  }
}

bool FormatStringConverter::emitIntegerArgument(
    ConversionSpecifier::Kind ArgKind, const Expr *Arg, unsigned ArgIndex,
    std::string &FormatSpec) {
  const clang::QualType &ArgType = Arg->getType();
  if (ArgType->isBooleanType()) {
    // std::format will print bool as either "true" or "false" by default,
    // but printf prints them as "0" or "1". Be compatible with printf by
    // requesting decimal output.
    FormatSpec.push_back('d');
  } else if (ArgType->isEnumeralType()) {
    // std::format will try to find a specialization to print the enum
    // (and probably fail), whereas printf would have just expected it to
    // be passed as its underlying type. However, printf will have forced
    // the signedness based on the format string, so we need to do the
    // same.
    if (const auto *ET = ArgType->getAs<EnumType>()) {
      if (const std::optional<std::string> MaybeCastType =
              castTypeForArgument(ArgKind, ET->getDecl()->getIntegerType()))
        ArgFixes.emplace_back(
            ArgIndex, (Twine("static_cast<") + *MaybeCastType + ">(").str());
      else
        return conversionNotPossible(
            (Twine("argument ") + Twine(ArgIndex) + " has unexpected enum type")
                .str());
    }
  } else if (CastMismatchedIntegerTypes &&
             !isMatchingSignedness(ArgKind, ArgType)) {
    // printf will happily print an unsigned type as signed if told to.
    // Even -Wformat doesn't warn for this. std::format will format as
    // unsigned unless we cast it.
    if (const std::optional<std::string> MaybeCastType =
            castTypeForArgument(ArgKind, ArgType))
      ArgFixes.emplace_back(
          ArgIndex, (Twine("static_cast<") + *MaybeCastType + ">(").str());
    else
      return conversionNotPossible(
          (Twine("argument ") + Twine(ArgIndex) + " cannot be cast to " +
           Twine(ArgKind == ConversionSpecifier::Kind::uArg ? "unsigned"
                                                            : "signed") +
           " integer type to match format"
           " specifier and StrictMode is enabled")
              .str());
  } else if (isRealCharType(ArgType) || !ArgType->isIntegerType()) {
    // Only specify integer if the argument is of a different type
    FormatSpec.push_back('d');
  }
  return true;
}

/// Append the corresponding standard format string type fragment to FormatSpec,
/// and store any argument fixes for later application.
/// @returns true on success, false on failure
bool FormatStringConverter::emitType(const PrintfSpecifier &FS, const Expr *Arg,
                                     std::string &FormatSpec) {
  ConversionSpecifier::Kind ArgKind = FS.getConversionSpecifier().getKind();
  switch (ArgKind) {
  case ConversionSpecifier::Kind::sArg:
    emitStringArgument(FS.getArgIndex() + ArgsOffset, Arg);
    break;
  case ConversionSpecifier::Kind::cArg:
    // The type must be "c" to get a character unless the type is exactly
    // char (whether that be signed or unsigned for the target.)
    if (!isRealCharType(Arg->getType()))
      FormatSpec.push_back('c');
    break;
  case ConversionSpecifier::Kind::dArg:
  case ConversionSpecifier::Kind::iArg:
  case ConversionSpecifier::Kind::uArg:
    if (!emitIntegerArgument(ArgKind, Arg, FS.getArgIndex() + ArgsOffset,
                             FormatSpec))
      return false;
    break;
  case ConversionSpecifier::Kind::pArg: {
    const clang::QualType &ArgType = Arg->getType();
    // std::format knows how to format void pointers and nullptrs
    if (!ArgType->isNullPtrType() && !ArgType->isVoidPointerType())
      ArgFixes.emplace_back(FS.getArgIndex() + ArgsOffset,
                            "static_cast<const void *>(");
    break;
  }
  case ConversionSpecifier::Kind::xArg:
    FormatSpec.push_back('x');
    break;
  case ConversionSpecifier::Kind::XArg:
    FormatSpec.push_back('X');
    break;
  case ConversionSpecifier::Kind::oArg:
    FormatSpec.push_back('o');
    break;
  case ConversionSpecifier::Kind::aArg:
    FormatSpec.push_back('a');
    break;
  case ConversionSpecifier::Kind::AArg:
    FormatSpec.push_back('A');
    break;
  case ConversionSpecifier::Kind::eArg:
    FormatSpec.push_back('e');
    break;
  case ConversionSpecifier::Kind::EArg:
    FormatSpec.push_back('E');
    break;
  case ConversionSpecifier::Kind::fArg:
    FormatSpec.push_back('f');
    break;
  case ConversionSpecifier::Kind::FArg:
    FormatSpec.push_back('F');
    break;
  case ConversionSpecifier::Kind::gArg:
    FormatSpec.push_back('g');
    break;
  case ConversionSpecifier::Kind::GArg:
    FormatSpec.push_back('G');
    break;
  default:
    // Something we don't understand
    return conversionNotPossible((Twine("argument ") +
                                  Twine(FS.getArgIndex() + ArgsOffset) +
                                  " has an unsupported format specifier")
                                     .str());
  }

  return true;
}

/// Append the standard format string equivalent of the passed PrintfSpecifier
/// to StandardFormatString and store any argument fixes for later application.
/// @returns true on success, false on failure
bool FormatStringConverter::convertArgument(const PrintfSpecifier &FS,
                                            const Expr *Arg,
                                            std::string &StandardFormatString) {
  // The specifier must have an associated argument
  assert(FS.consumesDataArgument());

  StandardFormatString.push_back('{');

  if (FS.usesPositionalArg()) {
    // std::format argument identifiers are zero-based, whereas printf ones
    // are one based.
    assert(FS.getPositionalArgIndex() > 0U);
    StandardFormatString.append(llvm::utostr(FS.getPositionalArgIndex() - 1));
  }

  // std::format format argument parts to potentially emit:
  // [[fill]align][sign]["#"]["0"][width]["."precision][type]
  std::string FormatSpec;

  // printf doesn't support specifying the fill character - it's always a
  // space, so we never need to generate one.

  emitAlignment(FS, FormatSpec);
  emitSign(FS, FormatSpec);
  emitAlternativeForm(FS, FormatSpec);

  if (FS.hasLeadingZeros())
    FormatSpec.push_back('0');

  emitFieldWidth(FS, FormatSpec);
  emitPrecision(FS, FormatSpec);
  maybeRotateArguments(FS);

  if (!emitType(FS, Arg, FormatSpec))
    return false;

  if (!FormatSpec.empty()) {
    StandardFormatString.push_back(':');
    StandardFormatString.append(FormatSpec);
  }

  StandardFormatString.push_back('}');
  return true;
}

/// Called for each format specifier by ParsePrintfString.
bool FormatStringConverter::HandlePrintfSpecifier(const PrintfSpecifier &FS,
                                                  const char *StartSpecifier,
                                                  unsigned SpecifierLen,
                                                  const TargetInfo &Target) {

  const size_t StartSpecifierPos = StartSpecifier - PrintfFormatString.data();
  assert(StartSpecifierPos + SpecifierLen <= PrintfFormatString.size());

  // Everything before the specifier needs copying verbatim
  assert(StartSpecifierPos >= PrintfFormatStringPos);

  appendFormatText(StringRef(PrintfFormatString.begin() + PrintfFormatStringPos,
                             StartSpecifierPos - PrintfFormatStringPos));

  const ConversionSpecifier::Kind ArgKind =
      FS.getConversionSpecifier().getKind();

  // Skip over specifier
  PrintfFormatStringPos = StartSpecifierPos + SpecifierLen;
  assert(PrintfFormatStringPos <= PrintfFormatString.size());

  FormatStringNeededRewriting = true;

  if (ArgKind == ConversionSpecifier::Kind::nArg) {
    // std::print doesn't do the equivalent of %n
    return conversionNotPossible("'%n' is not supported in format string");
  }

  if (ArgKind == ConversionSpecifier::Kind::PrintErrno) {
    // std::print doesn't support %m. In theory we could insert a
    // strerror(errno) parameter (assuming that libc has a thread-safe
    // implementation, which glibc does), but that would require keeping track
    // of the input and output parameter indices for position arguments too.
    return conversionNotPossible("'%m' is not supported in format string");
  }

  if (ArgKind == ConversionSpecifier::PercentArg) {
    StandardFormatString.push_back('%');
    return true;
  }

  const unsigned ArgIndex = FS.getArgIndex() + ArgsOffset;
  if (ArgIndex >= NumArgs) {
    // Argument index out of range. Give up.
    return conversionNotPossible(
        (Twine("argument index ") + Twine(ArgIndex) + " is out of range")
            .str());
  }

  return convertArgument(FS, Args[ArgIndex]->IgnoreImplicitAsWritten(),
                         StandardFormatString);
}

/// Called at the very end just before applying fixes to capture the last part
/// of the format string.
void FormatStringConverter::finalizeFormatText() {
  appendFormatText(
      StringRef(PrintfFormatString.begin() + PrintfFormatStringPos,
                PrintfFormatString.size() - PrintfFormatStringPos));
  PrintfFormatStringPos = PrintfFormatString.size();

  // It's clearer to convert printf("Hello\r\n"); to std::print("Hello\r\n")
  // than to std::println("Hello\r");
  // Use StringRef until C++20 std::string::ends_with() is available.
  const auto StandardFormatStringRef = StringRef(StandardFormatString);
  if (Config.AllowTrailingNewlineRemoval &&
      StandardFormatStringRef.ends_with("\\n") &&
      !StandardFormatStringRef.ends_with("\\\\n") &&
      !StandardFormatStringRef.ends_with("\\r\\n")) {
    UsePrintNewlineFunction = true;
    FormatStringNeededRewriting = true;
    StandardFormatString.erase(StandardFormatString.end() - 2,
                               StandardFormatString.end());
  }

  StandardFormatString.push_back('\"');
}

/// Append literal parts of the format text, reinstating escapes as required.
void FormatStringConverter::appendFormatText(const StringRef Text) {
  for (const char Ch : Text) {
    if (Ch == '\a')
      StandardFormatString += "\\a";
    else if (Ch == '\b')
      StandardFormatString += "\\b";
    else if (Ch == '\f')
      StandardFormatString += "\\f";
    else if (Ch == '\n')
      StandardFormatString += "\\n";
    else if (Ch == '\r')
      StandardFormatString += "\\r";
    else if (Ch == '\t')
      StandardFormatString += "\\t";
    else if (Ch == '\v')
      StandardFormatString += "\\v";
    else if (Ch == '\"')
      StandardFormatString += "\\\"";
    else if (Ch == '\\')
      StandardFormatString += "\\\\";
    else if (Ch == '{') {
      StandardFormatString += "{{";
      FormatStringNeededRewriting = true;
    } else if (Ch == '}') {
      StandardFormatString += "}}";
      FormatStringNeededRewriting = true;
    } else if (Ch < 32) {
      StandardFormatString += "\\x";
      StandardFormatString += llvm::hexdigit(Ch >> 4, true);
      StandardFormatString += llvm::hexdigit(Ch & 0xf, true);
    } else
      StandardFormatString += Ch;
  }
}

static std::string withoutCStrReplacement(const BoundNodes &CStrRemovalMatch,
                                          ASTContext &Context) {
  const auto *Arg = CStrRemovalMatch.getNodeAs<Expr>("arg");
  const auto *Member = CStrRemovalMatch.getNodeAs<MemberExpr>("member");
  const bool Arrow = Member->isArrow();
  return Arrow ? utils::fixit::formatDereference(*Arg, Context)
               : tooling::fixit::getText(*Arg, Context).str();
}

/// Called by the check when it is ready to apply the fixes.
void FormatStringConverter::applyFixes(DiagnosticBuilder &Diag,
                                       SourceManager &SM) {
  if (FormatStringNeededRewriting) {
    Diag << FixItHint::CreateReplacement(
        CharSourceRange::getTokenRange(FormatExpr->getBeginLoc(),
                                       FormatExpr->getEndLoc()),
        StandardFormatString);
  }

  // ArgCount is one less than the number of arguments to be rotated.
  for (auto [ValueArgIndex, ArgCount] : ArgRotates) {
    assert(ValueArgIndex < NumArgs);
    assert(ValueArgIndex > ArgCount);

    // First move the value argument to the right place. But if there's a
    // pending c_str() removal then we must do that at the same time.
    if (const auto CStrRemovalMatch =
            std::find_if(ArgCStrRemovals.cbegin(), ArgCStrRemovals.cend(),
                         [ArgStartPos = Args[ValueArgIndex]->getBeginLoc()](
                             const BoundNodes &Match) {
                           // This c_str() removal corresponds to the argument
                           // being moved if they start at the same location.
                           const Expr *CStrArg = Match.getNodeAs<Expr>("arg");
                           return ArgStartPos == CStrArg->getBeginLoc();
                         });
        CStrRemovalMatch != ArgCStrRemovals.end()) {
      const std::string ArgText =
          withoutCStrReplacement(*CStrRemovalMatch, *Context);
      assert(!ArgText.empty());

      Diag << FixItHint::CreateReplacement(
          Args[ValueArgIndex - ArgCount]->getSourceRange(), ArgText);

      // That c_str() removal is now dealt with, so we don't need to do it again
      ArgCStrRemovals.erase(CStrRemovalMatch);
    } else
      Diag << tooling::fixit::createReplacement(*Args[ValueArgIndex - ArgCount],
                                                *Args[ValueArgIndex], *Context);

    // Now shift down the field width and precision (if either are present) to
    // accommodate it.
    for (size_t Offset = 0; Offset < ArgCount; ++Offset)
      Diag << tooling::fixit::createReplacement(
          *Args[ValueArgIndex - Offset], *Args[ValueArgIndex - Offset - 1],
          *Context);

    // Now we need to modify the ArgFix index too so that we fix the right
    // argument. We don't need to care about the width and precision indices
    // since they never need fixing.
    for (auto &ArgFix : ArgFixes) {
      if (ArgFix.ArgIndex == ValueArgIndex)
        ArgFix.ArgIndex = ValueArgIndex - ArgCount;
    }
  }

  for (const auto &[ArgIndex, Replacement] : ArgFixes) {
    SourceLocation AfterOtherSide =
        Lexer::findNextToken(Args[ArgIndex]->getEndLoc(), SM, LangOpts)
            ->getLocation();

    Diag << FixItHint::CreateInsertion(Args[ArgIndex]->getBeginLoc(),
                                       Replacement, true)
         << FixItHint::CreateInsertion(AfterOtherSide, ")", true);
  }

  for (const auto &Match : ArgCStrRemovals) {
    const auto *Call = Match.getNodeAs<CallExpr>("call");
    const std::string ArgText = withoutCStrReplacement(Match, *Context);
    if (!ArgText.empty())
      Diag << FixItHint::CreateReplacement(Call->getSourceRange(), ArgText);
  }
}
} // namespace clang::tidy::utils
