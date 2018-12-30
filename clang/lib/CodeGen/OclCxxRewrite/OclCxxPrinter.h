//===- OclCxxPrinter.h - OCLC++ type/name printer & mangler     -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//
// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXPRINTER_H
#define CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXPRINTER_H

#define OCLCXXREWRITE_PRINTER_USE_LLVM_STREAMS 0

#include "OclCxxDemanglerResult.h"

#include <list>
#include <memory>
#include <type_traits>
#include <utility>

#if OCLCXXREWRITE_PRINTER_USE_LLVM_STREAMS
  #include "llvm/Support/raw_ostream.h"
#else
  #include <iosfwd>
#endif


// Headers for Itanium encoding traits (for separation to another file).
#include "OclCxxDemangler.h"
#include "OclCxxDemanglerResult.h"

#include <cassert>
#include <memory>
#include <stack>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#if OCLCXXREWRITE_PRINTER_USE_LLVM_STREAMS
  #include "llvm/Support/raw_ostream.h"
#else
  #include <sstream>
#endif


namespace oclcxx {
namespace adaptation {
namespace printer {

#if OCLCXXREWRITE_PRINTER_USE_LLVM_STREAMS
  using PrintOStreamT  = llvm::raw_ostream;
  using StringOStreamT = llvm::raw_string_ostream;
#else
  using PrintOStreamT  = std::ostream;
  using StringOStreamT = std::ostringstream;
#endif

// -----------------------------------------------------------------------------
// HELPERS FOR ENCODER / PRINTER
// -----------------------------------------------------------------------------
// String formatting / conversion

/// \brief Trims white-spaces from beginning and end of the string.
///
/// Only ASCII white-spaces are removed.
///
/// \tparam StringT Type of input string.
/// \param  Str     String value to trim.
/// \return String with leading and ending white-spaces stripped.
template <typename StringT>
inline StringT trim(const StringT &Str) {
  auto TrimBeginPos = Str.find_first_not_of(" \t\v\n\f\r");
  auto TrimEndPos = Str.find_last_not_of(" \t\v\n\f\r");

  if (TrimBeginPos != StringT::npos && TrimEndPos != StringT::npos)
    return Str.substr(TrimBeginPos, TrimEndPos - TrimBeginPos + 1);
  return StringT();
}

/// \brief Converts integral number to hexadecimal representation.
///
/// \tparam StringT    Type of string returned by conversion function.
///                    It must support construction from "const char *".
/// \tparam ShowSign   Indicates that hex value shoud be presented with sign.
///                    When the parameter is true, the form for presentation
///                    is {sign}{hex from absolute value}.
///                    When parameter is false, the form for presentation
///                    is {hex of 2^n least unsigned congruent with value}.
/// \tparam ShowPrefix Indicates that additional "0x" prefix should be added.
///                    If sign is show the prefix is inserted between sign
///                    and value.
/// \tparam ZeroFill   Indicates that hexadecimal representation should be
///                    zero-filled (from left) to length needed to present
///                    specific integral type.
/// \tparam LowerCase  Indicates that hexadecimal digits A-F are presented
///                    as lower-case letters.
/// \param  Val        Number to convert.
/// \return            String with hexadecimal representation of number.
template <typename StringT,
          bool ShowSign = false,
          bool ShowPrefix = false,
          bool ZeroFill = false,
          bool LowerCase = false,
          typename IntT>
inline typename std::enable_if<std::is_integral<IntT>::value, StringT>::type
asHex(const IntT &Val) {
  using UIntT = typename std::make_unsigned<IntT>::type;

  char Buffer[sizeof(Val) * 2 + 4]; // Place for <sign>0x and NULL-terminator.
  char *BufferPos = Buffer + sizeof(Buffer);
  *--BufferPos = '\0';

#ifdef _MSC_VER
  // NOTE: This is for removal of annoying non-C++ compliant error when
  //       using negate operation with unsigned type. The behavior in C++11
  //       is well-defined, but on VS with SDL checks it will generate error
  //       here. This makes harder to use template code, so we
  //       can disable this warning.
  #pragma warning(push)
  #pragma warning(disable: 4146)
#endif
  UIntT UVal = !ShowSign || Val >= 0 ? static_cast<UIntT>(Val)
                                     : -static_cast<UIntT>(Val);
#ifdef _MSC_VER
  #pragma warning(pop)
#endif

  do {
    char Digit = (UVal & 0x0F) + '0';
    Digit = Digit > '9' ? Digit + ((LowerCase ? 'a' : 'A')  - '9' - 1)
                        : Digit;
    *--BufferPos = Digit;
    UVal >>= 4;
  }
  while (UVal > 0);

  if (ZeroFill) {
    char *BufferFillPos = Buffer + 3; // Leave place for <sign>0x.
    while (BufferPos > BufferFillPos)
      *--BufferPos = '0';
  }
  if (ShowPrefix) {
    *--BufferPos = 'x';
    *--BufferPos = '0';
  }
  if (ShowSign)
    *--BufferPos = Val >= 0 ? '+' : '-';

  return StringT(BufferPos);
}

/// \brief Converts integral number to base-36 representation.
///
/// \tparam StringT   Type of string returned by conversion function.
///                   It must support construction from "const char *".
/// \tparam LowerCase Indicates that base-36 digits A-Z are presented
///                   as lower-case letters.
/// \param  Val       Number to convert (threated as unsigned).
/// \return           String with base-36 representation of number.
template <typename StringT,
          bool LowerCase = false,
          typename IntT>
inline typename std::enable_if<std::is_integral<IntT>::value, StringT>::type
asBase36(const IntT &Val) {
  using UIntT = typename std::make_unsigned<IntT>::type;

  char Buffer[sizeof(Val) * 8 / 5 + 2]; // Place for NULL-terminator.
  char *BufferPos = Buffer + sizeof(Buffer);
  *--BufferPos = '\0';

  UIntT UVal = static_cast<UIntT>(Val);

  do {
    char Digit = (UVal % 36) + '0';
    Digit = Digit > '9' ? Digit + ((LowerCase ? 'a' : 'A')  - '9' - 1)
                        : Digit;
    *--BufferPos = Digit;
    UVal /= 36;
  }
  while (UVal > 0);

  return StringT(BufferPos);
}

// -----------------------------------------------------------------------------
// Stream wrappers / helpers

/// Placeholder output stream.
///
/// Stream adapter which is capable of creating child streams. Child stream
/// can be attached to any position inside parent stream and its content
/// will be inserted / spliced at that point.
/// It is not thread-safe.
/// Stream adapter delays any flush until flush() is explicitly called
/// on root stream or root stream is being destroyed.
///
/// \tparam StringT        String storage type. The type should have interface
///                        compatible with std::basic_string.
/// \tparam OStreamT       Output stream type. Its interface should be
///                        compatible with either std::basic_ostream or
///                        llvm::raw_ostream.
///                        OStreamT must support operator<< with StringT value.
/// \tparam StringOStreamT String-based output stream. Its interface should be
///                        compatible with either std::basic_ostringstream or
///                        llvm::raw_string_ostream.
///                        Type must be publicly non-ambiguously derived from
///                        OStreamT (reference-convertible) and constructible
///                        from lvalue reference to StringT.
template <typename StringT, typename OStreamT, typename StringOStreamT>
class PlaceholderOStream {
  static_assert(std::is_convertible<StringOStreamT &, OStreamT &>::value,
                "StringOStreamT: Selected output string stream class is not"
                "convertible to selected base stream class (OStreamT).");
  static_assert(std::is_constructible<StringOStreamT, StringT &>::value,
                "StringOStreamT: Selected output string stream class cannot"
                "be constructed from selected string class (StringT).");

  /// \brief Type of collection which can contain output parts and support
  ///        splicing (that does not invalidate iterators).
  using PlaceholderColT = std::list<StringT>;
  // FIXME: Move back to const_iterator once we go above GCC 4.7.
  //        Currently it does not support list<T>::insert with const_iterator.
  //        Restore also all cbegin/cend uses.
  /// \brief Type of splice position used for splicing in PlaceholderColT.
  using PlaceholderPosT = typename PlaceholderColT::iterator;

  /// \brief Represents all data in placeholder stream shared with children
  ///        streams.
  struct StreamData {
    /// \brief Flushes stream data to wrapped stream and increments flush
    /// counter.
    ///
    /// Due to use of flush counter, the number of flushes (in root)
    /// is limited to maximum value of unsigned long long
    /// (usually at least 2^64-1 times).
    void flush() {
      // If there is nothing to flush, nothing needs to be done.
      if (Placeholders.empty())
        return;

      // Failure of flush if flush counter overflown.
      if (FlushCounter + 1 <= FlushCounter) {
        assert(false && "Flush counter overflow");
        return;
      }

      for (const auto &Placeholder : Placeholders) {
        if (!Placeholder.empty())
          OStream << Placeholder;
      }
      Placeholders.clear();

      ++FlushCounter;
    }

    /// \brief Constructs stream data with wrapped output stream.
    ///
    /// Lifetime of wrapped stream is not extended.
    ///
    /// \param Out Output stream to wrap.
    explicit StreamData(OStreamT &Out)
      : OStream(Out), FlushCounter(0) {}


    /// Wrapped output stream by current placeholder stream.
    /// Only most parent (root) placeholder stream fills it.
    OStreamT &OStream;
    /// Collection of placeholder strings which will be sent to output stream.
    PlaceholderColT Placeholders;
    /// Flush counter to indicate whether child streams are being used after
    /// flush in root stream. If child counter is lower than shared one,
    /// the child placeholder position is invalid.
    unsigned long long FlushCounter;
  };


  /// \brief Inserts string at back of placeholder specified by current stream.
  ///
  /// For root stream, the value is inserted at the end of placeholders.
  /// For child streams, the value is inserted before placeholder end position
  /// determined at creation of the child stream.
  /// If child stream is invalid (after flush() of the root stream), the
  /// function does nothing.
  ///
  /// \tparam StringArgT Type of string argument (std::forward pattern).
  /// \param  Str        String value to insert.
  /// \param  AllowEmpty Indicates that empty string will be also inserted.
  /// \return            Position of inserted placeholder, or end() if value
  ///                    was not inserted.
  template <typename StringArgT>
  PlaceholderPosT pushBackAtPlaceholder(StringArgT &&Str,
                                        bool AllowEmpty = true) {
    if (!isValid() || (!AllowEmpty && Str.empty()))
      return Data->Placeholders.end();
    return Data->Placeholders.insert(PlaceholderEnd,
                                     std::forward<StringArgT>(Str));
  }

  /// \brief Inserts string at front of placeholder specified by current stream.
  ///
  /// For root stream, the value is inserted at the front of placeholders.
  /// For child streams, the value is inserted after placeholder front position
  /// determined at creation of the child stream.
  /// If child stream is invalid (after flush() of the root stream), the
  /// function does nothing.
  ///
  /// \tparam StringArgT Type of string argument (std::forward pattern).
  /// \param  Str        String value to insert.
  /// \param  AllowEmpty Indicates that empty string will be also inserted.
  /// \return            Position of inserted placeholder, or end() if value
  ///                    was not inserted.
  template <typename StringArgT>
  PlaceholderPosT pushFrontAtPlaceholder(StringArgT &&Str,
                                         bool AllowEmpty = true) {
    if (!isValid() || (!AllowEmpty && Str.empty()))
      return Data->Placeholders.end();
    auto ChildBegin = PlaceholderBegin;
    return Data->Placeholders.insert(IsRoot ? Data->Placeholders.begin()
                                            : ++ChildBegin,
                                     std::forward<StringArgT>(Str));
  }

public:
  /// \brief Indicates that stream is valid and writes to it will not be
  ///        discarded.
  bool isValid() const {
    if (IsRoot)
      return true;
    return FlushCounter >= Data->FlushCounter;
  }

  /// \brief Indicates that stream is pure.
  ///
  /// Stream is pure when it is valid and no modification was done on it since
  /// its creation or flush of root stream. Pure stream is empty and contains
  /// no placeholder substreams (placeholder stream buffers are empty).
  bool isPure() const {
    if (!isValid())
      return false;
    if (IsRoot)
      return Data->Placeholders.empty();
    auto ChildBegin = PlaceholderBegin;
    return (++ChildBegin == PlaceholderEnd);
  }

  /// \brief Returns current content of the stream as string.
  StringT str() const {
    if (!isValid())
      return StringT();

    // NOTE: String storage to support llvm::raw_string_ostream.
    StringT Str;
    // ReSharper disable once CppLocalVariableMightNotBeInitialized
    StringOStreamT StrOut(Str);

    auto ChildBegin = PlaceholderBegin;
    for (auto B = IsRoot ? Data->Placeholders.begin() : ++ChildBegin;
         B != PlaceholderEnd; ++B) {
      if (!B->empty())
        StrOut << *B;
    }

    return StrOut.str();
  }

  /// \brief Flushes root placeholder stream.
  ///
  /// If used on root stream, flushes content to wrapped output stream.
  /// Child streams are invalidated and all writes to these streams will
  /// be discarded.
  /// If used on any child stream, it does nothing.
  ///
  /// The root stream supports limited number of flushes (at least 2^64-1).
  void flush() {
    if (IsRoot)
      Data->flush();
  }

  /// \brief Creates child placeholder stream from current stream at
  ///        current front position.
  PlaceholderOStream createPrefixChildStream() {
    auto PosEnd = pushFrontAtPlaceholder(StringT());
    auto PosBegin = pushFrontAtPlaceholder(StringT());
    return PlaceholderOStream(*this, PosBegin, PosEnd);
  }

  /// \brief Creates child placeholder stream from current stream at
  ///        current back position.
  PlaceholderOStream createChildStream() {
    auto PosBegin = pushBackAtPlaceholder(StringT());
    auto PosEnd = pushBackAtPlaceholder(StringT());
    return PlaceholderOStream(*this, PosBegin, PosEnd);
  }


  /// \brief Constructs placeholder stream from other output stream.
  ///
  /// Lifetime of wrapped stream is not extended.
  ///
  /// \param Out Output stream to wrap.
  explicit PlaceholderOStream(OStreamT &Out)
    : Data(std::make_shared<StreamData>(Out)),
      PlaceholderBegin(Data->Placeholders.begin()), // useless in root stream
      PlaceholderEnd(Data->Placeholders.end()),     // valid in root stream
      FlushCounter(0), IsRoot(true) {}

private:
  /// \brief Constructs child placeholder stream stream from parent stream.
  ///
  /// \param Parent              Parent placeholder stream.
  /// \param PlaceholderBegin    Position in parent stream after which
  ///                            element from child stream will be inserted.
  /// \param PlaceholderEnd      Position in parent stream before which
  ///                            element from child stream will be inserted.
  PlaceholderOStream(PlaceholderOStream &Parent,
                     const PlaceholderPosT &PlaceholderBegin,
                     const PlaceholderPosT &PlaceholderEnd)
    : Data(Parent.Data),
      PlaceholderBegin(PlaceholderBegin),
      PlaceholderEnd(PlaceholderEnd),
      FlushCounter(Parent.Data->FlushCounter), IsRoot(false) {}

public:
  /// \brief Destructor of the current stream.
  ///
  /// Stream is flushed before destruction.
  ~PlaceholderOStream() {
    flush();
  }


  /// \brief Formatted output operator.
  ///
  /// \tparam ValueT Type of value to output.
  /// \param  Value  Value to output.
  /// \return        Current stream.
  template <typename ValueT>
  PlaceholderOStream &operator <<(const ValueT &Value) {
    // NOTE: String storage to support llvm::raw_string_ostream.
    StringT Str;
    // ReSharper disable once CppLocalVariableMightNotBeInitialized
    StringOStreamT StrOut(Str);

    StrOut << Value;
    pushBackAtPlaceholder(StrOut.str(), false);

    return *this;
  }


private:
  /// Shared data for all placeholder streams.
  std::shared_ptr<StreamData> Data;

  /// Position in parent stream after which elements from current stream
  /// will be added (if parent exists).
  ///
  /// Must be after Data (see order of initialization).
  PlaceholderPosT PlaceholderBegin;
  /// Position in parent stream before which elements from current stream
  /// will be added (if parent exists).
  ///
  /// Must be after Data (see order of initialization).
  PlaceholderPosT PlaceholderEnd;
  /// Separate flush counter. If counter is set to value lower than in shared
  /// data, the stream has got invalid position.
  unsigned long long FlushCounter;
  /// Indicates that placeholder stream is root stream (most parent).
  bool IsRoot;
};

// -----------------------------------------------------------------------------
// ENCODER / PRINTER TRAITS
// -----------------------------------------------------------------------------

/// Status from print / encode functions used in printing.
enum EncodeResult {
  ER_Failure,           ///< Encoding of name / part of name failed.
  ER_Success,           ///< Encoding of name / part of name succeeded.
  ER_SuccessSubstitute, ///< Encoding of name / part of name succeeded
                        ///< and resulting string should be substituted or
                        ///< registered as substitution (if possible).
  ER_SuccessNoMangle    ///< Encoding of name / part of name succeeded,
                        ///< but name was simple and required no mangling.
};

/// Status from adaptation functions used in printing.
enum AdaptResult {
  AR_Failure,  ///< Adaptation failed.
  AR_Adapt,    ///< Adaptation succeeded and node was changed during adaptation.
  AR_NoAdapt   ///< Adaptation succeeded, but adapted node is the same as
               ///< node before adaptation.
};

/// Printing / encoding traits (C++-like).
///
/// Set of types / handlers which instructs printing or encoding of demangler
/// result structures.
struct CxxLikeEncodeTraits {
  /// Type of string used as storage for partial printer / encoder results.
  using StringT         = std::string;
  /// Type of output stream used by printer / encoder.
  using OStreamT        = printer::PrintOStreamT;
  /// Type of string-based output stream used by printer / encoder.
  ///
  /// Type of string used as backing storage must be the same as StringT.
  /// The stream type must be derived from OStreamT.
  /// The stream must be constructible from reference to StringT.
  using StringOStreamT  = printer::StringOStreamT;
  /// Wrapper stream that wraps string-based output stream.
  ///
  /// Wrapper stream allows to add custom behavior to output streams used in
  /// encode() functions.
  /// The custom wrapper must be at least CopyConstructible.
  ///
  /// Use lvalue reference to StringOStreamT: "StringOStreamT &" and implement
  /// createWrappedStream() as identity function if wrapper is not used.
  using OStreamWrapperT = PlaceholderOStream<StringT, OStreamT, StringOStreamT>;


  /// Creates custom wrapper for string-based output stream.
  ///
  /// Use lvalue reference to StringOStreamT: "StringOStreamT &" and implement
  /// createWrappedStream() as identity function if wrapper is not used.
  ///
  /// \param Out Wrapped output stream.
  /// \return    New instance of wrapper.
  ///            The custom wrapper must be at least CopyConstructible.
  OStreamWrapperT createStreamWrapper(StringOStreamT &Out);

  /// Performs a final processing on result string.
  ///
  /// \param  Result Result generated from encodeResult() function.
  /// \return        Corrected / processed result.
  StringT processResult(const StringT &Result);


  /// Encodes result node into result string.
  ///
  /// \param Out   Wrapped output stream.
  /// \param Value Value to encode (node of demangler result).
  /// \return      Value indicating state of encodeResult() operation.
  EncodeResult encodeResult(OStreamWrapperT &Out,
                            const std::shared_ptr<const DmngRsltNode> &Value);

  /// Encodes demangler's result into result string.
  ///
  /// \param Out    Wrapped output stream.
  /// \param Result Demangler result to encode.
  /// \return       Value indicating state of encodeResult() operation.
  EncodeResult encodeResult(OStreamWrapperT &Out, const DmngRslt &Result);

private:
  // Non-node types.
  EncodeResult encode(OStreamWrapperT &Out, const DmngRsltVendorQual &Value);
  EncodeResult encode(OStreamWrapperT &Out, const DmngRsltTArg &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltTArg> &Value);
  EncodeResult encode(OStreamWrapperT &Out, const DmngRsltAdjustOffset &Value);
  EncodeResult encode(OStreamWrapperT &Out, const DmngCvrQuals &Value);
  EncodeResult encode(OStreamWrapperT &Out, const DmngRefQuals &Value);

  // Non-node abstract bases.
  EncodeResult encodeVendorQuals(OStreamWrapperT &Out,
                                 const DmngRsltVendorQualsBase &Value);
  EncodeResult encodeTArgs(OStreamWrapperT &Out,
                           const DmngRsltTArgsBase &Value);
  EncodeResult encodeSignatureParams(OStreamWrapperT &Out,
                                     const DmngRsltSignatureTypesBase &Value);
  EncodeResult encodeSignatureReturn(OStreamWrapperT &Out,
                                     const DmngRsltSignatureTypesBase &Value);
  EncodeResult encodeNameParts(OStreamWrapperT &Out,
                               const DmngRsltNamePartsBase &Value);

  // Nodes (expression).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltExpr> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltDecltypeExpr> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltTParamExpr> &Value,
                      bool ReverseAssembly = false);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltPrimaryExpr> &Value);

  // Nodes (type).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltBuiltinType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltFuncType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltTypeNameType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltArrayVecType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltPtr2MmbrType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltTParamType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltDecltypeType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltQualType> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltQualGrpType> &Value);

  // Nodes (name).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltName> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltOrdinaryName> &Value);
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltSpecialName> &Value);

  // Nodes (name parts).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltNamePart> &Value) {
    return encode(Out, Value, nullptr);
  }
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltNamePart> &Value,
                      const std::shared_ptr<const DmngRsltNamePart> &PrevPart);
  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltOpNamePart> &Value);
  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltCtorDtorNamePart> &Value,
      const std::shared_ptr<const DmngRsltNamePart> &PrevPart = nullptr);
  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltSrcNamePart> &Value);
  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltUnmTypeNamePart> &Value);
  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltTParamNamePart> &Value);
  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltDecltypeNamePart> &Value);
};

// -----------------------------------------------------------------------------

/// \def OCLCXX_IMPT_ENCODE(...)
/// Invokes encode function overload for specified node with short circuiting
/// when encode fails.
///
/// Parameters are the same as for encode() function.
#define OCLCXX_IMPT_ENCODE(...)                                                \
do {                                                                           \
  if (encode(__VA_ARGS__) == ER_Failure)                                       \
    return ER_Failure;                                                         \
} while(false)

/// \def OCLCXX_IMPT_ADAPT(...)
/// Invokes adaptNode function for specified node with short circuiting
/// when adaptation takes place.
///
/// Parameters are the same as for adaptNode() function.
#define OCLCXX_IMPT_ADAPT(...)                                                 \
do {                                                                           \
  const auto &AdaptRet = adaptNode(__VA_ARGS__);                               \
  if (AdaptRet.first)                                                          \
    return AdaptRet.second;                                                    \
} while(false)

/// Adaptation traits that do no do any adaptation.
struct NoAdaptationTraits {
  /// Dummmy adapt function for lookup resolution.
  void adapt() const;
};

/// Printing / encoding traits (Itanium-mangling).
///
/// Set of types / handlers which instructs printing or encoding of demangler
/// result structures.
///
/// Adaptation traits is simple inheritable class / structure which meets
/// DefaultConstructible, CopyConstructible, CopyAssignable, Destructible
/// requirements (optionally: MoveConstructible, MoveAssignable).
/// Adaptation traits must implement at least one adapt() method with
/// following / similar signature:
///   std::pair<AdaptResult, ResultT>
///     adapt(const std::shared_ptr<const NodeT> &Node) const
/// where:
///  * NodeT is type of demangler result node.
///  * ResultT is one of:
///  ** Type of demangler result node (in which case it is encoded as one).
///  ** std::string (in which case it is encoded as source name).
///  ** std::nullptr_t / empty shared pointer (in which case it is
///     not encoded at all).
/// Node parameter identifies node to adapt and return value conatains pair
/// which first, enum value indicates node's adaptation status and
/// second value contains node after adaptation (only processed if first value
/// is AR_Adapt.
/// adapt() methods can be declared as non-const (it is recommended though).
///
///
/// \tparam AdaptationTraits     Adaptation traits (described in details).
///                              If non-class type is specified, the fall-back
///                              (non-adapting) traits are used.
/// \tparam UseSubstitutions     Indicates that substitution should be used.
///                              (Itanium-mangling substitutions).
/// \tparam ExpandTArgs          Indicates that template parameters should
///                              be expanded to referred template arguments.
/// \tparam ExpandVariadicPacks  Indicates that variadic template parameter
///                              pack expansion should be expanded to
///                              non-template form.
/// \tparam SimpleNamesUnmangled Indicates that simple data names (non-local
///                              non-template non-nested non-member non-function
///                              non-special non-qualified names) should be
///                              printed without mangling (as they were
///                              extern "C").
///                              The option is applied only when printing
///                              DmngRslt.
//
// TODO: Handle substitutions in the future (currently template parameter
//       is ignored).
template <typename AdaptationTraits = void,
          bool UseSubstitutions = true,
          bool ExpandTArgs = false,
          bool ExpandVariadicPacks = false,
          bool SimpleNamesUnmangled = false>
struct ItaniumEncodeTraits
  : private std::conditional<std::is_class<AdaptationTraits>::value,
                             AdaptationTraits, NoAdaptationTraits>::type {

  /// Type of string used as storage for partial printer / encoder results.
  using StringT         = std::string;
  /// Type of output stream used by printer / encoder.
  using OStreamT        = printer::PrintOStreamT;
  /// Type of string-based output stream used by printer / encoder.
  ///
  /// Type of string used as backing storage must be the same as StringT.
  /// The stream type must be derived from OStreamT.
  /// The stream must be constructible from reference to StringT.
  using StringOStreamT  = printer::StringOStreamT;
  /// Wrapper stream that wraps string-based output stream.
  ///
  /// Wrapper stream allows to add custom behavior to output streams used in
  /// encode() functions.
  /// The custom wrapper must be at least CopyConstructible.
  ///
  /// Use lvalue reference to StringOStreamT: "StringOStreamT &" and implement
  /// createWrappedStream() as identity function if wrapper is not used.
  using OStreamWrapperT = StringOStreamT &;


  /// Creates custom wrapper for string-based output stream.
  ///
  /// Use lvalue reference to StringOStreamT: "StringOStreamT &" and implement
  /// createWrappedStream() as identity function if wrapper is not used.
  ///
  /// \param Out Wrapped output stream.
  /// \return    New instance of wrapper.
  ///            The custom wrapper must be at least CopyConstructible.
  OStreamWrapperT createStreamWrapper(StringOStreamT &Out) {
    return Out;
  }

  /// Performs a final processing on result string.
  ///
  /// \param  Result Result generated from encodeResult() function.
  /// \return        Corrected / processed result.
  StringT processResult(const StringT &Result) {
    return Result;
  }


  /// Encodes result node into result string.
  ///
  /// \param Out   Wrapped output stream.
  /// \param Value Value to encode (node of demangler result).
  /// \return      Value indicating state of encodeResult() operation.
  EncodeResult encodeResult(OStreamWrapperT &Out,
                            const std::shared_ptr<const DmngRsltNode> &Value) {
    if (Value == nullptr)
      return ER_Failure;

    switch (Value->getNodeKind()) {
    case DNDK_Name:
      return encode(Out, Value->getAs<DNDK_Name>());
    case DNDK_NamePart:
      return encode(Out, Value->getAs<DNDK_NamePart>());
    case DNDK_Type:
      return encode(Out, Value->getAs<DNDK_Type>());
    case DNDK_Expr:
      return encode(Out, Value->getAs<DNDK_Expr>());
    case DNDK_NameParts:
      // ReSharper disable once CppUnreachableCode
      assert(false &&
             "Printer does not support printing of name parts helper.");
      return ER_Failure;
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current node kind.");
      return ER_Failure;
    }
  }

  /// Encodes demangler's result into result string.
  ///
  /// \param Out    Wrapped output stream.
  /// \param Result Demangler result to encode.
  /// \return       Value indicating state of encodeResult() operation.
  EncodeResult encodeResult(OStreamWrapperT &Out, const DmngRslt &Result) {
    if (Result.isFailed() || Result.getName() == nullptr)
      return ER_Failure;

    if (SimpleNamesUnmangled) {
      // NOTE: Helper string to support llvm::raw_string_ostream.
      StringT EncodingStr;
      StringOStreamT EncodingOut(EncodingStr);

      auto Status = encode(EncodingOut, Result.getName(), true);
      if (Status == ER_Failure)
        return ER_Failure;
      if (Status != ER_SuccessNoMangle)
        Out << "_Z";
      Out << EncodingOut.str();
      return Status;
    }

    Out << "_Z";
    return encode(Out, Result.getName());
  }


private:
  /// \brief Adaptation traits class used as base for current print traits.
  using MyBaseT = typename std::conditional<
                    std::is_class<AdaptationTraits>::value,
                    AdaptationTraits, NoAdaptationTraits>::type;

public:
  /// \brief Creates instance of printing traits (Itanium-mangling).
  ItaniumEncodeTraits()
    : PackExpansionLevel(0) {}

  /// \brief Copy constructor.
  ItaniumEncodeTraits(const ItaniumEncodeTraits &Other) = default;
  /// \brief Copy assignment operator.
  ItaniumEncodeTraits &operator =(const ItaniumEncodeTraits &Other) = default;

  /// \brief Move constructor.
  ItaniumEncodeTraits(ItaniumEncodeTraits &&Other)
    : MyBaseT(std::move(Other)),
      PackExpansions(std::move(Other.PackExpansions)),
      PackExpansionLevel(Other.PackExpansionLevel) {}

  /// \brief Move assignment operator.
  ItaniumEncodeTraits &operator =(ItaniumEncodeTraits &&Other) {
    if (this == &Other)
      return *this;

    MyBaseT::operator =(std::move(Other));
    PackExpansions = std::move(Other.PackExpansions);
    PackExpansionLevel = Other.PackExpansionLevel;

    return *this;
  }


private:
  /// \brief Position of template argument pack expansion.
  struct PackExpansionPosition {
    PackExpansionPosition(
      const std::shared_ptr<const DmngRsltTArg> &DmngRsltTArg,
      StringT::size_type InsertPos, bool IsUnwrappedExprAllowed)
      : Pack(DmngRsltTArg),
        InsertPos(InsertPos),
        IsUnwrappedExprAllowed(IsUnwrappedExprAllowed) {}

    /// Template argument pack to be expanded.
    std::shared_ptr<const DmngRsltTArg> Pack;
    /// Position where expanded parameter should be inserted.
    StringT::size_type InsertPos;
    /// Indicates that in current context of template parameter expansion
    /// parameter can be expanded to expression (non-primary) without wrapping
    /// it in X-E section.
    bool IsUnwrappedExprAllowed;
  };

// -----------------------------------------------------------------------------
// Adaptation

  // Introduce into scope all needed functions from adaptation base.
  using MyBaseT::adapt;

  /// \brief Fall-back node adaptation function.
  ///
  /// Performs no adaptation.
  ///
  /// \return Pair with enum value and adapted node (if node was adapted).
  ///         Enum value indicates whether adaptation took place and changed
  ///         node.
  template <typename ... ArgsT>
  std::pair<AdaptResult, std::nullptr_t> adapt(ArgsT && ...) const {
    return std::make_pair(AR_NoAdapt, nullptr);
  }

  /// \brief Adapts node if possible.
  ///
  /// Method tries to adapt node using current adaptation traits. If node
  /// was adapted, the adapted node / adapted value is encoded and result
  /// of encoding is returned. If node was not adapted, the value indicating
  /// that adaptation was not done is returned.
  ///
  /// \tparam NodeT Type of result node to adapt.
  ///
  /// \param Out                    Output stream for encoding.
  /// \param Node                   Node to adapt.
  /// \param IsUnwrappedExprAllowed Indicates that expression not wrapped in X-E
  ///                               are allowed in current context.
  /// \param IsSimpleName           Indicates that node is a part of simple
  ///                               name.
  /// \return Pair of boolean and enumeration value.
  ///         First, boolean value indicates that adaptation took place.
  ///         If adaptation was done, the second value indicates result
  ///         of encoding of adapted node / adapted value.
  template <typename NodeT>
  std::pair<bool, EncodeResult>
  adaptNode(OStreamWrapperT &Out,
            const std::shared_ptr<const NodeT> &Node,
            bool IsUnwrappedExprAllowed = false,
            bool SimpleName = false) {

    auto AdaptedNode = adapt(Node);
    if (AdaptedNode.first == AR_Failure)
      return std::make_pair(true, ER_Failure);
    if (AdaptedNode.first == AR_NoAdapt)
      return std::make_pair(false, ER_Success);

    return std::make_pair(true, encodeAdaptedNode(Out, AdaptedNode.second,
                                                  IsUnwrappedExprAllowed,
                                                  SimpleName));
  }

  /// \brief Encoding delegation (most of nodes types).
  ///
  /// Used when adapted node is of result node type. Ignores X-E wrapping
  /// option. Ignores simple name option.
  template <typename NodeT>
  typename std::enable_if<!std::is_base_of<DmngRsltExpr, NodeT>::value &&
                          !std::is_base_of<DmngRsltName, NodeT>::value &&
                          !std::is_base_of<DmngRsltNamePart, NodeT>::value,
                          EncodeResult>::type
  encodeAdaptedNode(OStreamWrapperT &Out,
                    const std::shared_ptr<const NodeT> &Node,
                    bool, bool) {
    if (Node == nullptr)
      return ER_Success;
    return encodeResult(Out, Node);
  }

  /// \brief Encoding delegation (possible X-E wrapped node types).
  ///
  /// Used when adapted node is of result node type. Takes into consideration
  /// X-E wrapping option.
  EncodeResult
  encodeAdaptedNode(OStreamWrapperT &Out,
                    const std::shared_ptr<const DmngRsltTArg> &Node,
                    bool IsUnwrappedExprAllowed, bool) {
    if (Node == nullptr)
      return ER_Success;
    return encode(Out, Node, IsUnwrappedExprAllowed);
  }

  /// \brief Encoding delegation (possible X-E wrapped node types).
  ///
  /// Used when adapted node is of result node type. Takes into consideration
  /// X-E wrapping option.
  EncodeResult
  encodeAdaptedNode(OStreamWrapperT &Out,
                    const std::shared_ptr<const DmngRsltExpr> &Node,
                    bool IsUnwrappedExprAllowed, bool) {
    if (Node == nullptr)
      return ER_Success;
    return encode(Out, Node, IsUnwrappedExprAllowed);
  }

  /// \brief Encoding delegation (string).
  ///
  /// Special case: If string is returned from adaptation, it is encoded
  ///               as source name in Itanium mangling.
  EncodeResult
  encodeAdaptedNode(OStreamWrapperT &Out, const std::string &Node, bool,
                    bool SimpleName) {
    if (Node.empty())
      return ER_Failure;

    if (SimpleNamesUnmangled && SimpleName) {
      Out << Node;
      return ER_SuccessNoMangle;
    }
    Out << Node.length() << Node;
    return ER_Success;
  }

  /// \brief Encoding delegation (names).
  ///
  /// Used when adapted node is of result node type. Takes into consideration
  /// whether simple name should be left unmangled.
  EncodeResult
  encodeAdaptedNode(OStreamWrapperT &Out,
                    const std::shared_ptr<const DmngRsltName> &Node,
                    bool, bool SimpleName) {
    if (Node == nullptr)
      return ER_Success;
    return encode(Out, Node, SimpleName);
  }

  /// \brief Encoding delegation (name parts).
  ///
  /// Used when adapted node is of result node type. Takes into consideration
  /// whether simple name should be left unmangled.
  EncodeResult
  encodeAdaptedNode(OStreamWrapperT &Out,
                    const std::shared_ptr<const DmngRsltNamePart> &Node,
                    bool, bool SimpleName) {
    if (Node == nullptr)
      return ER_Success;
    return encode(Out, Node, SimpleName);
  }

  /// \brief Encoding delegation (nullptr).
  ///
  /// Special case: If nullptr is returned from adaptation, the node is not
  ///               encoded at all.
  EncodeResult
  encodeAdaptedNode(OStreamWrapperT &, std::nullptr_t, bool, bool) {
    return ER_Success;
  }

  /// \brief Encoding delegation (boolean).
  ///
  /// Special case: If boolean is returned from adaptation, the node is not
  ///               encoded at all, but the boolean value indicates whether
  ///               we should fail adaptation/encoding.
  EncodeResult encodeAdaptedNode(OStreamWrapperT &, bool Success, bool, bool) {
    if (Success)
      return ER_Success;
    return ER_Failure;
  }

// -----------------------------------------------------------------------------
// Printing / encoding

  /// \brief Indicates that name is nested name.
  bool isNestedName(const DmngRsltOrdinaryName &Value) const {
    // Name with qualifiers is always nested.
    if (Value.getCvrQuals() != DCVQ_None || Value.getRefQuals() != DRQ_None ||
        !Value.getVendorQuals().empty())
      return true;

    // 3 or more parts are always nested.
    if (Value.getParts().size() >= 3)
      return true;
    // Single unqualified part is never nested.
    if (Value.getParts().size() <= 1)
      return false;

    // 2 parts (it depends whether first part is ::std:: namespace;
    // if it is in ::std:: namespace, the name is not nested name).
    const auto &NsPart = Value.getParts().front();
    if (NsPart == nullptr || NsPart->getPartKind() != DNPK_Source ||
        NsPart->isTemplate())
      return true;
    auto &&NsSrcPart = NsPart->getAs<DNPK_Source>();
    return NsSrcPart->getSourceName() != "std";
  }

  /// \brief Indicates that name is simple data/type name.
  ///
  /// This method only provide initial testing of name node. It also requires
  /// testing in name part node using isSimpleNamePart.
  ///
  /// Simple name is non-local non-template non-nested non-member non-function
  /// non-special non-qualified name.
  bool isSimpleName(const DmngRsltOrdinaryName &Value) const {
    if (Value.isFunction())
      return false; // function
    if (Value.isLocal())
      return false; // local
    if (Value.getDefaultValueParamRIdx() >= 0)
      return false; // member (parameter default value scope)
    if (Value.isStringLiteral())
      return false; // special (string literal)

    if (Value.getCvrQuals() != DCVQ_None || Value.getRefQuals() != DRQ_None ||
        !Value.getVendorQuals().empty())
      return false; // qualified

    if (Value.getParts().size() != 1)
      return false; // nested

    return true;
  }

  /// \brief Indicates that name is simple data/type name.
  ///
  /// This method only provide final testing of name part node. It also requires
  /// previous testing in name node using isSimpleName.
  ///
  /// Simple name is non-local non-template non-nested non-member non-function
  /// non-special non-qualified name.
  bool isSimpleNamePart(const DmngRsltSrcNamePart &Value) const {
    if (Value.isTemplate())
      return false; // template
    if (Value.isDataMember())
      return false; // special

    return true;
  }


  // Non-node types.
  template <typename IntT>
  typename std::enable_if<std::is_integral<IntT>::value, void>::type
  encodeNumber(OStreamWrapperT &Out, const IntT &Number) {
    using UIntT = typename std::make_unsigned<IntT>::type;

  #ifdef _MSC_VER
    // NOTE: This is for removal of annoying non-C++ compliant error when
    //       using negate operation with unsigned type. The behavior in C++11
    //       is well-defined, but on VS with SDL checks it will generate error
    //       here. This makes harder to use template code, so we
    //       can disable this warning.
    #pragma warning(push)
    #pragma warning(disable: 4146)
  #endif
    UIntT UNumber = Number >= 0 ? static_cast<UIntT>(Number)
                                : -static_cast<UIntT>(Number);
  #ifdef _MSC_VER
    #pragma warning(pop)
  #endif

    if (Number < 0)
      Out << "n";
    Out << UNumber;
  }

  template <typename UIntT>
  typename std::enable_if<std::is_integral<UIntT>::value, void>::type
  encodeDiscriminator(OStreamWrapperT &Out, const UIntT &Number) {
    if (Number > 10)
      Out << "__" << Number - 1 << "_";
    else if (Number > 0)
      Out << "_" << Number - 1;
  }

  EncodeResult encode(OStreamWrapperT &Out, const DmngRsltVendorQual &Value) {
    Out << "U" << Value.getName().length() << Value.getName();

    return encodeTArgs(Out, Value);
  }

  EncodeResult encode(OStreamWrapperT &Out, const DmngRsltTArg &Value,
                      bool IsUnwrappedExprAllowed) {
    if (Value.isType())
      return encode(Out, Value.getType());

    if (Value.isExpression()) {
      const auto &Expr = Value.getExpression();

      if (Expr->getKind() == DXK_Primary || IsUnwrappedExprAllowed)
        return encode(Out, Value.getExpression());

      Out << "X";
      OCLCXX_IMPT_ENCODE(Out, Value.getExpression(), true);
      Out << "E";

      return ER_Success;
    }

    if (Value.isPack()) {
      Out << "J";
      for (const auto &TArg : Value.getPack()) {
        OCLCXX_IMPT_ENCODE(Out, TArg, false);
      }
      Out << "E";

      return ER_Success;
    }

    return ER_Failure;
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltTArg> &Value,
                      bool IsUnwrappedExprAllowed) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, IsUnwrappedExprAllowed);

    return encode(Out, *Value, IsUnwrappedExprAllowed);
  }

  EncodeResult encode(OStreamWrapperT &Out, const DmngRsltAdjustOffset &Value) {
    if (Value.isVirtual()) {
      Out << "v";
      encodeNumber(Out, Value.getBaseOffset());
      Out << "_";
      encodeNumber(Out, Value.getVCallOffset());
    }
    else {
      Out << "h";
      encodeNumber(Out, Value.getBaseOffset());
    }

    return ER_Success;
  }

  EncodeResult encode(OStreamWrapperT &Out, const DmngCvrQuals &Value) {
    if (Value & DCVQ_Restrict)
      Out << "r";
    if (Value & DCVQ_Volatile)
      Out << "V";
    if (Value & DCVQ_Const)
      Out << "K";

    return ER_Success;
  }

  EncodeResult encode(OStreamWrapperT &Out, const DmngRefQuals &Value) {
    switch (Value) {
    case DRQ_None:      break;
    case DRQ_LValueRef: Out << "R"; break;
    case DRQ_RValueRef: Out << "O"; break;
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current reference qualifier.");
      return ER_Failure;
    }

    return ER_Success;
  }


  // Non-node abstract bases.
  EncodeResult encodeVendorQuals(OStreamWrapperT &Out,
                                 const DmngRsltVendorQualsBase &Value) {
    // NOTE: Value.getAsQuals() is already encoded in vendor-extended qualifiers
    //       Only raw qualifiers will be encoded.
    for (const auto &VQual : Value.getVendorQuals()) {
      OCLCXX_IMPT_ENCODE(Out, VQual);
    }

    return ER_Success;
  }

  EncodeResult encodeTArgs(OStreamWrapperT &Out,
                           const DmngRsltTArgsBase &Value) {
    if (Value.isTemplate()) {
      Out << "I";
      for (const auto &TArg : Value.getTemplateArgs()) {
        OCLCXX_IMPT_ENCODE(Out, TArg, false);
      }
      Out << "E";
    }

    return ER_Success;
  }

  EncodeResult encodeSignatureParams(OStreamWrapperT &Out,
                                     const DmngRsltSignatureTypesBase &Value) {
    if (Value.getSignatureTypes().empty()) {
      Out << "v";
      return ER_Success;
    }

    // NOTE: Helper string to support llvm::raw_string_ostream.
    StringT SignatureStr;
    StringOStreamT SignatureOut(SignatureStr);

    for (const auto &ParamType : Value.getParamTypes()) {
      OCLCXX_IMPT_ENCODE(SignatureOut, ParamType);
    }

    SignatureStr = SignatureOut.str(); // Handle empty signatures.
    if (SignatureStr.empty())
      Out << "v";
    else
      Out << SignatureStr;

    return ER_Success;
  }

  EncodeResult encodeSignatureReturn(OStreamWrapperT &Out,
                                     const DmngRsltSignatureTypesBase &Value) {
    if (Value.hasReturnType())
      return encode(Out, Value.getReturnType());

    return ER_Success;
  }

  EncodeResult encodeNameParts(OStreamWrapperT &Out,
                               const DmngRsltNamePartsBase &Value,
                               bool SimpleName = false) {
    // TODO: Substitution...
    for (const auto &NamePart : Value.getParts()) {
      // TODO: Name part without template added to substitution.
      // TODO: All except last part.
      auto Status = encode(Out, NamePart, SimpleName);
      if (Status == ER_Failure)
        return ER_Failure;
      if (SimpleNamesUnmangled && Status == ER_SuccessNoMangle)
        return ER_SuccessNoMangle;

      if (NamePart->isDataMember())
        Out << "M";
    }

    return ER_Success;
  }


  // Nodes (expression).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltExpr> &Value,
                      bool IsUnwrappedExprAllowed = false) {
    if (Value == nullptr)
      return ER_Failure;

    switch (Value->getKind()) {
    case DXK_Decltype:
      return encode(Out, Value->getAs<DXK_Decltype>());
    case DXK_TemplateParam:
      return encode(Out, Value->getAs<DXK_TemplateParam>(),
                    IsUnwrappedExprAllowed);
    case DXK_Primary:
      return encode(Out, Value->getAs<DXK_Primary>());
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current node kind.");
      return ER_Failure;
    }
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltDecltypeExpr> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    Out << (Value->isSimple() ? "Dt" : "DT");
    OCLCXX_IMPT_ENCODE(Out, Value->getExpression(), true);
    Out << "E";

    return ER_Success;
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltTParamExpr> &Value,
                      bool IsUnwrappedExprAllowed = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, IsUnwrappedExprAllowed);

    if (ExpandTArgs && Value->getReferredTArg() != nullptr &&
        !Value->getReferredTArg()->isPack())
      return encode(Out, Value->getReferredTArg(), IsUnwrappedExprAllowed);

    if (ExpandVariadicPacks && Value->getReferredTArg() != nullptr &&
        Value->getReferredTArg()->isPack()) {
      PackExpansions.push(PackExpansionPosition {Value->getReferredTArg(),
                                                 Out.str().length(),
                                                 IsUnwrappedExprAllowed});
      return ER_Success;
    }

    if (Value->getReferredTArgIdx() > 0)
      Out << "T" << Value->getReferredTArgIdx() - 1 << "_";
    else
      Out << "T_";

    return ER_Success;
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltPrimaryExpr> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    Out << "L";
    if (Value->isExternalName()) {
      Out << "_Z";
      OCLCXX_IMPT_ENCODE(Out, Value->getExternalName());
    } else if (Value->isLiteral()) {
      OCLCXX_IMPT_ENCODE(Out, Value->getLiteralType());

      switch (Value->getContentType()) {
      case DmngRsltPrimaryExpr::Void:
        break;
      case DmngRsltPrimaryExpr::UInt:
        encodeNumber(Out, Value->getContentAsUInt());
        break;
      case DmngRsltPrimaryExpr::SInt:
        encodeNumber(Out, Value->getContentAsSInt());
        break;
      case DmngRsltPrimaryExpr::Bool:
        encodeNumber(Out, static_cast<int>(Value->getContentAsBool())); // 0/1
        break;
      default:
        // ReSharper disable once CppUnreachableCode
        assert(false && "Printer does not support literal content type.");
        return ER_Failure;
      }
    }
    Out << "E";

    return ER_Success;
  }


  // Nodes (type).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltType> &Value) {
    if (Value == nullptr)
      return ER_Failure;

    switch (Value->getKind()) {
    case DTK_Builtin:
      return encode(Out, Value->getAs<DTK_Builtin>());
    case DTK_Function:
      return encode(Out, Value->getAs<DTK_Function>());
    case DTK_TypeName:
      return encode(Out, Value->getAs<DTK_TypeName>());
    case DTK_Array:
      return encode(Out, Value->getAs<DTK_Array>());
    case DTK_Vector:
      return encode(Out, Value->getAs<DTK_Vector>());
    case DTK_PointerToMember:
      return encode(Out, Value->getAs<DTK_PointerToMember>());
    case DTK_TemplateParam:
      return encode(Out, Value->getAs<DTK_TemplateParam>());
    case DTK_Decltype:
      return encode(Out, Value->getAs<DTK_Decltype>());
    case DTK_Pointer:
      return encode(Out, Value->getAs<DTK_Pointer>());
    case DTK_LValueRef:
      return encode(Out, Value->getAs<DTK_LValueRef>());
    case DTK_RValueRef:
      return encode(Out, Value->getAs<DTK_RValueRef>());
    case DTK_C2000Complex:
      return encode(Out, Value->getAs<DTK_C2000Complex>());
    case DTK_C2000Imaginary:
      return encode(Out, Value->getAs<DTK_C2000Imaginary>());
    case DTK_PackExpansion:
      return encode(Out, Value->getAs<DTK_PackExpansion>());
    case DTK_QualGroup:
      return encode(Out, Value->getAs<DTK_QualGroup>());
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current node kind.");
      return ER_Failure;
    }
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltBuiltinType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    if (Value->isVendorBuiltinType())
      Out << "u" << Value->getVendorName().length() << Value->getVendorName();
    else
      Out << getEncFixedBuiltinTypeName(Value->getBuiltinType());

    return ER_Success;
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltFuncType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    if (encodeVendorQuals(Out, *Value) == ER_Failure)
      return ER_Failure;
    OCLCXX_IMPT_ENCODE(Out, Value->getCvrQuals());
    // TODO: Add transactional qualifier in the future (Dx).
    Out << "F";
    if (Value->isExternC())
      Out << "Y";
    if (encodeSignatureReturn(Out, *Value) == ER_Failure)
      return ER_Failure;
    if (encodeSignatureParams(Out, *Value) == ER_Failure)
      return ER_Failure;
    OCLCXX_IMPT_ENCODE(Out, Value->getRefQuals());
    Out << "E";

    return ER_Success;
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltTypeNameType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    switch (Value->getElaboration()) {
    case DTNK_None:            break;
    case DTNK_ElaboratedClass: Out << "Ts"; break;
    case DTNK_ElaboratedUnion: Out << "Tu"; break;
    case DTNK_ElaboratedEnum:  Out << "Te"; break;
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current elaboration.");
      return ER_Failure;
    }
    return encode(Out, Value->getTypeName());
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltArrayVecType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    Out << (Value->getKind() == DTK_Vector ? "Dv" : "A");
    if (Value->isSizeSpecified()) {
      if (Value->getSizeExpr() != nullptr) {
        OCLCXX_IMPT_ENCODE(Out, Value->getSizeExpr(), true);
      }
      else
        Out << Value->getSize();
    }
    Out << "_";
    return encode(Out, Value->getElemType());
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltPtr2MmbrType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    Out << "M";
    OCLCXX_IMPT_ENCODE(Out, Value->getClassType());
    return encode(Out, Value->getMemberType());
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltTParamType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    OCLCXX_IMPT_ENCODE(Out, Value->getTemplateParam());
    return encodeTArgs(Out, *Value);
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltDecltypeType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    return encode(Out, Value->getDecltype());
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltQualType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    if (ExpandVariadicPacks && Value->getKind() == DTK_PackExpansion) {
      ++PackExpansionLevel;

      // NOTE: Helper string to support llvm::raw_string_ostream.
      StringT ExpPackStr;
      StringOStreamT ExpPackOut(ExpPackStr);

      OCLCXX_IMPT_ENCODE(ExpPackOut, Value->getInnerType());

      if (PackExpansionLevel <= 0 ||
          PackExpansionLevel != PackExpansions.size()) {
        // ReSharper disable once CppUnreachableCode
        assert(false && "There is no corresponding template parameter"
                        " with pack to expand by pack expansion qualifier.");
        return ER_Failure;
      }

      auto ExpPos = PackExpansions.top();
      ExpPackStr = ExpPackOut.str();
      if (ExpPos.InsertPos > ExpPackStr.length()) {
        // ReSharper disable once CppUnreachableCode
        assert(false && "Pack expansion position is invalid.");
        return ER_Failure;
      }

      StringT ExpPackPrefix = ExpPackStr.substr(0, ExpPos.InsertPos);
      StringT ExpPackSuffix = ExpPackStr.substr(ExpPos.InsertPos);
      for (const auto &PackElem : ExpPos.Pack->getPack()) {
        Out << ExpPackPrefix;
        OCLCXX_IMPT_ENCODE(Out, PackElem, ExpPos.IsUnwrappedExprAllowed);
        Out << ExpPackSuffix;
      }

      PackExpansions.pop();
      --PackExpansionLevel;

      return ER_Success;
    }

    switch (Value->getKind()) {
    case DTK_Pointer:
      Out << "P";
      break;
    case DTK_LValueRef:
      Out << "R";
      break;
    case DTK_RValueRef:
      Out << "O";
      break;
    case DTK_C2000Complex:
      Out << "C";
      break;
    case DTK_C2000Imaginary:
      Out << "G";
      break;
    case DTK_PackExpansion:
      Out << "Dp";
      break;
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support qualifier.");
      return ER_Failure;
    }

    return encode(Out, Value->getInnerType());
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltQualGrpType> &Value) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value);

    if (encodeVendorQuals(Out, *Value) == ER_Failure)
      return ER_Failure;
    OCLCXX_IMPT_ENCODE(Out, Value->getCvrQuals());
    return encode(Out, Value->getInnerType());
  }


  // Nodes (name).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltName> &Value,
                      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;

    switch (Value->getKind()) {
    case DNK_Ordinary:
      return encode(Out, Value->getAs<DNK_Ordinary>(), SimpleName);
    case DNK_Special:
      return encode(Out, Value->getAs<DNK_Special>(), SimpleName);
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current node kind.");
      return ER_Failure;
    }
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltOrdinaryName> &Value,
      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    if (SimpleNamesUnmangled && SimpleName && isSimpleName(*Value))
      return encodeNameParts(Out, *Value, true);

    if (Value->isLocal()) {
      Out << "Z";
      OCLCXX_IMPT_ENCODE(Out, Value->getLocalScope());
      Out << "E";
    }

    if (Value->getDefaultValueParamRIdx() >= 0) {
      Out << "d";
      if (Value->getDefaultValueParamRIdx() > 0)
        Out << Value->getDefaultValueParamRIdx() - 1;
      Out << "_";
    }
    else if (Value->isStringLiteral()) {
      Out << "s";
      encodeDiscriminator(Out, Value->getInLocalScopeIdx());
      return ER_Success;
    }

    bool IsNestedName = isNestedName(*Value);
    if (IsNestedName)
      Out << "N";
    if (encodeVendorQuals(Out, *Value) == ER_Failure)
      return ER_Failure;
    OCLCXX_IMPT_ENCODE(Out, Value->getCvrQuals());
    OCLCXX_IMPT_ENCODE(Out, Value->getRefQuals());
    if (encodeNameParts(Out, *Value) == ER_Failure)
      return ER_Failure;
    if (IsNestedName)
      Out << "E";

    if (Value->getDefaultValueParamRIdx() < 0)
      encodeDiscriminator(Out, Value->getInLocalScopeIdx());

    if (Value->isFunction()) {
      if (encodeSignatureReturn(Out, *Value) == ER_Failure)
        return ER_Failure;
      if (encodeSignatureParams(Out, *Value) == ER_Failure)
        return ER_Failure;
    }

    return ER_Success;
  }

  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltSpecialName> &Value,
                      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    // TODO: Add transactional stubs (GTt) in the future.
    switch (Value->getSpecialKind()) {
    case DSNK_VirtualTable:
      Out << "TV";
      return encode(Out, Value->getRelatedType());
    case DSNK_VirtualTableTable:
      Out << "TT";
      return encode(Out, Value->getRelatedType());
    case DSNK_TypeInfoStruct:
      Out << "TI";
      return encode(Out, Value->getRelatedType());
    case DSNK_TypeInfoNameString:
      Out << "TS";
      return encode(Out, Value->getRelatedType());
    case DSNK_VirtualThunk:
      if (Value->getReturnAdjustment().isZero()) {
        Out << "T";
        OCLCXX_IMPT_ENCODE(Out, Value->getThisAdjustment());
      }
      else {
        Out << "Tc";
        OCLCXX_IMPT_ENCODE(Out, Value->getThisAdjustment());
        OCLCXX_IMPT_ENCODE(Out, Value->getReturnAdjustment());
      }
      return encode(Out, Value->getOrigin());
    case DSNK_GuardVariable:
      Out << "GV";
      return encode(Out, Value->getRelatedObject());
    case DSNK_LifeExtTemporary:
      Out << "GR";
      OCLCXX_IMPT_ENCODE(Out, Value->getRelatedObject());
      if (Value->getId() > 0)
        Out << asBase36<StringT>(Value->getId() - 1);
      Out << "_";
      return ER_Success;
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current special name kind.");
      return ER_Failure;
    }
  }


  // Nodes (name parts).
  EncodeResult encode(OStreamWrapperT &Out,
                      const std::shared_ptr<const DmngRsltNamePart> &Value,
                      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;

    switch (Value->getPartKind()) {
    case DNPK_Operator:
      return encode(Out, Value->getAs<DNPK_Operator>(), SimpleName);
    case DNPK_Constructor:
      return encode(Out, Value->getAs<DNPK_Constructor>(), SimpleName);
    case DNPK_Destructor:
      return encode(Out, Value->getAs<DNPK_Destructor>(), SimpleName);
    case DNPK_Source:
      return encode(Out, Value->getAs<DNPK_Source>(), SimpleName);
    case DNPK_UnnamedType:
      return encode(Out, Value->getAs<DNPK_UnnamedType>(), SimpleName);
    case DNPK_TemplateParam:
      return encode(Out, Value->getAs<DNPK_TemplateParam>(), SimpleName);
    case DNPK_Decltype:
      return encode(Out, Value->getAs<DNPK_Decltype>(), SimpleName);
    case DNPK_DataMember:
      return encode(Out, Value->getAs<DNPK_DataMember>(), SimpleName);
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current name part kind.");
      return ER_Failure;
    }
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltOpNamePart> &Value,
      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    if (Value->isConversionOperator()) {
      Out << "cv";
      OCLCXX_IMPT_ENCODE(Out, Value->getConvertTargetType());
    }
    else if (Value->isLiteralOperator()) {
      Out << "li" << Value->getLiteralOperatorSuffix().length()
          << Value->getLiteralOperatorSuffix();
    }
    else if (Value->isVendorOperator()) {
      if (Value->getVendorOperatorArity() < 0 ||
          Value->getVendorOperatorArity() > 9) {
        // ReSharper disable once CppUnreachableCode
        assert(false && "Printer does not support current operator arity.");
        return ER_Failure;
      }

      Out << "v" << Value->getVendorOperatorArity()
          << Value->getVendorOperatorName().length()
          << Value->getVendorOperatorName();
    }
    else
      Out << getEncFixedOperatorName(Value->getNameCode());

    return encodeTArgs(Out, *Value);
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltCtorDtorNamePart> &Value,
      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    Out << (Value->getPartKind() == DNPK_Constructor ? "C" : "D");
    switch (Value->getType()) {
    case DCDT_BaseObj:
      Out << "2";
      break;
    case DCDT_CompleteObj:
      Out << "1";
      break;
    case DCDT_DynMemObj:
      Out << (Value->getPartKind() == DNPK_Constructor ? "3" : "0");
      break;
    default:
      // ReSharper disable once CppUnreachableCode
      assert(false && "Printer does not support current ctor/dtor type.");
      return ER_Failure;
    }

    return encodeTArgs(Out, *Value);
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltSrcNamePart> &Value,
      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    if (SimpleNamesUnmangled && SimpleName && isSimpleNamePart(*Value)) {
      Out << Value->getSourceName();
      return ER_SuccessNoMangle;
    }
    Out << Value->getSourceName().length() << Value->getSourceName();
    return encodeTArgs(Out, *Value);
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltUnmTypeNamePart> &Value,
      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    if (Value->isClosure()) {
      Out << "Ul";
      if (encodeSignatureParams(Out, *Value) == ER_Failure)
        return ER_Failure;
      Out << "E";
    }
    else
      Out << "Ut";

    if (Value->getId() > 0)
      Out << Value->getId() - 1;
    Out << "_";

    return encodeTArgs(Out, *Value);
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltTParamNamePart> &Value,
      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    OCLCXX_IMPT_ENCODE(Out, Value->getTemplateParam());
    return encodeTArgs(Out, *Value);
  }

  EncodeResult encode(
      OStreamWrapperT &Out,
      const std::shared_ptr<const DmngRsltDecltypeNamePart> &Value,
      bool SimpleName = false) {
    if (Value == nullptr)
      return ER_Failure;
    OCLCXX_IMPT_ADAPT(Out, Value, false, SimpleName);

    OCLCXX_IMPT_ENCODE(Out, Value->getDecltype());
    return encodeTArgs(Out, *Value);
  }


  /// Stack of pack expansions (for variadic pack expansion).
  std::stack<PackExpansionPosition> PackExpansions;
  /// Level of pack expansion analized during encode().
  unsigned PackExpansionLevel;
};

#undef OCLCXX_CLPT_ENCODE
#undef OCLCXX_IMPT_ADAPT

// -----------------------------------------------------------------------------
// ENCODER / PRINTER IMPLEMENTATION
// -----------------------------------------------------------------------------

/// \brief Prints encoded demangler result or its part.
///
/// State of printer is set to initial.
///
/// \param Out   Output stream where print result should be stored.
/// \param Value Value to print (demangler's result or its part (node)).
/// \return      Pair with first argument indicating that operation succeeded
///              and second containing state of printer (encoding state).
template <typename EncodeTraitsT, typename ValueT>
std::pair<bool, EncodeTraitsT> print(typename EncodeTraitsT::OStreamT &Out,
           const ValueT &Value) {
  EncodeTraitsT EncTraits;
  auto Success = print(EncTraits, Out, Value);

  return std::make_pair(Success, std::move(EncTraits));
}

/// \brief Prints encoded demangler result or its part.
///
/// This function allows to generate partial encoding based on state from
/// previous prints.
///
/// \param State Encoding state from previous prints.
/// \param Out   Output stream where print result should be stored.
/// \param Value Value to print (demangler's result or its part (node)).
/// \return      Value indicating that operation succeeded.
template <typename EncodeTraitsT, typename ValueT>
bool print(EncodeTraitsT &State, typename EncodeTraitsT::OStreamT &Out,
           const ValueT &Value) {
  using StringT         = typename EncodeTraitsT::StringT;
  using StringOStreamT  = typename EncodeTraitsT::StringOStreamT;
  using OStreamWrapperT = typename EncodeTraitsT::OStreamWrapperT;

  EncodeTraitsT EncTraits(State);
  // NOTE: Helper string to support llvm::raw_string_ostream.
  StringT Result;
  StringOStreamT ResultOut(Result);
  {
    OStreamWrapperT ResultWrapper = EncTraits.createStreamWrapper(ResultOut);
    if (EncTraits.encodeResult(ResultWrapper, Value) == ER_Failure)
      return false;
  }

  Out << EncTraits.processResult(ResultOut.str());
  return true;
}


} // printer
} // adaptation
} // oclcxx

#endif // CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXPRINTER_H
