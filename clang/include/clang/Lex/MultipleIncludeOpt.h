//===--- MultipleIncludeOpt.h - Header Multiple-Include Optzn ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the MultipleIncludeOpt interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_MULTIPLEINCLUDEOPT_H
#define LLVM_CLANG_MULTIPLEINCLUDEOPT_H

namespace clang {
class IdentifierInfo;

/// \brief Implements the simple state machine that the Lexer class uses to
/// detect files subject to the 'multiple-include' optimization.
///
/// The public methods in this class are triggered by various
/// events that occur when a file is lexed, and after the entire file is lexed,
/// information about which macro (if any) controls the header is returned.
class MultipleIncludeOpt {
  /// ReadAnyTokens - This is set to false when a file is first opened and true
  /// any time a token is returned to the client or a (non-multiple-include)
  /// directive is parsed.  When the final \#endif is parsed this is reset back
  /// to false, that way any tokens before the first \#ifdef or after the last
  /// \#endif can be easily detected.
  bool ReadAnyTokens;

  /// ReadAnyTokens - This is set to false when a file is first opened and true
  /// any time a token is returned to the client or a (non-multiple-include)
  /// directive is parsed.  When the final #endif is parsed this is reset back
  /// to false, that way any tokens before the first #ifdef or after the last
  /// #endif can be easily detected.
  bool DidMacroExpansion;

  /// TheMacro - The controlling macro for a file, if valid.
  ///
  const IdentifierInfo *TheMacro;
public:
  MultipleIncludeOpt() {
    ReadAnyTokens = false;
    DidMacroExpansion = false;
    TheMacro = 0;
  }

  /// Invalidate - Permanently mark this file as not being suitable for the
  /// include-file optimization.
  void Invalidate() {
    // If we have read tokens but have no controlling macro, the state-machine
    // below can never "accept".
    ReadAnyTokens = true;
    TheMacro = 0;
  }

  /// getHasReadAnyTokensVal - This is used for the \#ifndef hande-shake at the
  /// top of the file when reading preprocessor directives.  Otherwise, reading
  /// the "ifndef x" would count as reading tokens.
  bool getHasReadAnyTokensVal() const { return ReadAnyTokens; }

  // If a token is read, remember that we have seen a side-effect in this file.
  void ReadToken() { ReadAnyTokens = true; }

  /// ExpandedMacro - When a macro is expanded with this lexer as the current
  /// buffer, this method is called to disable the MIOpt if needed.
  void ExpandedMacro() { DidMacroExpansion = true; }

  /// \brief Called when entering a top-level \#ifndef directive (or the
  /// "\#if !defined" equivalent) without any preceding tokens.
  ///
  /// Note, we don't care about the input value of 'ReadAnyTokens'.  The caller
  /// ensures that this is only called if there are no tokens read before the
  /// \#ifndef.  The caller is required to do this, because reading the \#if
  /// line obviously reads in in tokens.
  void EnterTopLevelIFNDEF(const IdentifierInfo *M) {
    // If the macro is already set, this is after the top-level #endif.
    if (TheMacro)
      return Invalidate();

    // If we have already expanded a macro by the end of the #ifndef line, then
    // there is a macro expansion *in* the #ifndef line.  This means that the
    // condition could evaluate differently when subsequently #included.  Reject
    // this.
    if (DidMacroExpansion)
      return Invalidate();

    // Remember that we're in the #if and that we have the macro.
    ReadAnyTokens = true;
    TheMacro = M;
  }

  /// \brief Invoked when a top level conditional (except \#ifndef) is found.
  void EnterTopLevelConditional() {
    // If a conditional directive (except #ifndef) is found at the top level,
    // there is a chunk of the file not guarded by the controlling macro.
    Invalidate();
  }

  /// \brief Called when the lexer exits the top-level conditional.
  void ExitTopLevelConditional() {
    // If we have a macro, that means the top of the file was ok.  Set our state
    // back to "not having read any tokens" so we can detect anything after the
    // #endif.
    if (!TheMacro) return Invalidate();

    // At this point, we haven't "read any tokens" but we do have a controlling
    // macro.
    ReadAnyTokens = false;
  }

  /// \brief Once the entire file has been lexed, if there is a controlling
  /// macro, return it.
  const IdentifierInfo *GetControllingMacroAtEndOfFile() const {
    // If we haven't read any tokens after the #endif, return the controlling
    // macro if it's valid (if it isn't, it will be null).
    if (!ReadAnyTokens)
      return TheMacro;
    return 0;
  }
};

}  // end namespace clang

#endif
