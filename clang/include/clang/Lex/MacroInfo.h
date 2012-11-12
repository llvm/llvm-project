//===--- MacroInfo.h - Information about #defined identifiers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MacroInfo interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_MACROINFO_H
#define LLVM_CLANG_MACROINFO_H

#include "clang/Lex/Token.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include <cassert>

namespace clang {
  class Preprocessor;

/// MacroInfo - Each identifier that is \#define'd has an instance of this class
/// associated with it, used to implement macro expansion.
class MacroInfo {
  //===--------------------------------------------------------------------===//
  // State set when the macro is defined.

  /// Location - This is the place the macro is defined.
  SourceLocation Location;
  /// EndLocation - The location of the last token in the macro.
  SourceLocation EndLocation;
  /// \brief The location where the macro was #undef'd, or an invalid location
  /// for macros that haven't been undefined.
  SourceLocation UndefLocation;
  /// \brief Previous definition, the identifier of this macro was defined to,
  /// or NULL.
  MacroInfo *PreviousDefinition;

  /// Arguments - The list of arguments for a function-like macro.  This can be
  /// empty, for, e.g. "#define X()".  In a C99-style variadic macro, this
  /// includes the \c __VA_ARGS__ identifier on the list.
  IdentifierInfo **ArgumentList;
  unsigned NumArguments;

  /// \brief The location at which this macro was either explicitly exported
  /// from its module or marked as private.
  ///
  /// If invalid, this macro has not been explicitly given any visibility.
  SourceLocation VisibilityLocation;
  
  /// \brief This is the list of tokens that the macro is defined to.
  SmallVector<Token, 8> ReplacementTokens;

  /// \brief Length in characters of the macro definition.
  mutable unsigned DefinitionLength;
  mutable bool IsDefinitionLengthCached : 1;

  /// \brief True if this macro is a function-like macro, false if it
  /// is an object-like macro.
  bool IsFunctionLike : 1;

  /// IsC99Varargs - True if this macro is of the form "#define X(...)" or
  /// "#define X(Y,Z,...)".  The __VA_ARGS__ token should be replaced with the
  /// contents of "..." in an invocation.
  bool IsC99Varargs : 1;

  /// IsGNUVarargs -  True if this macro is of the form "#define X(a...)".  The
  /// "a" identifier in the replacement list will be replaced with all arguments
  /// of the macro starting with the specified one.
  bool IsGNUVarargs : 1;

  /// IsBuiltinMacro - True if this is a builtin macro, such as __LINE__, and if
  /// it has not yet been redefined or undefined.
  bool IsBuiltinMacro : 1;

  /// \brief True if this macro was loaded from an AST file.
  bool IsFromAST : 1;

  /// \brief Whether this macro changed after it was loaded from an AST file.
  bool ChangedAfterLoad : 1;
  
private:
  //===--------------------------------------------------------------------===//
  // State that changes as the macro is used.

  /// IsDisabled - True if we have started an expansion of this macro already.
  /// This disables recursive expansion, which would be quite bad for things
  /// like \#define A A.
  bool IsDisabled : 1;

  /// IsUsed - True if this macro is either defined in the main file and has
  /// been used, or if it is not defined in the main file.  This is used to
  /// emit -Wunused-macros diagnostics.
  bool IsUsed : 1;

  /// AllowRedefinitionsWithoutWarning - True if this macro can be redefined
  /// without emitting a warning.
  bool IsAllowRedefinitionsWithoutWarning : 1;

  /// \brief Must warn if the macro is unused at the end of translation unit.
  bool IsWarnIfUnused : 1;
   
  /// \brief Whether the macro has public (when described in a module).
  bool IsPublic : 1;

  /// \brief Whether the macro definition is currently "hidden".
  /// Note that this is transient state that is never serialized to the AST
  /// file.
  bool IsHidden : 1;

  /// \brief Whether the definition of this macro is ambiguous, due to
  /// multiple definitions coming in from multiple modules.
  bool IsAmbiguous : 1;

   ~MacroInfo() {
    assert(ArgumentList == 0 && "Didn't call destroy before dtor!");
  }

public:
  MacroInfo(SourceLocation DefLoc);
  MacroInfo(const MacroInfo &MI, llvm::BumpPtrAllocator &PPAllocator);
  
  /// FreeArgumentList - Free the argument list of the macro, restoring it to a
  /// state where it can be reused for other devious purposes.
  void FreeArgumentList() {
    ArgumentList = 0;
    NumArguments = 0;
  }

  /// Destroy - destroy this MacroInfo object.
  void Destroy() {
    FreeArgumentList();
    this->~MacroInfo();
  }

  /// getDefinitionLoc - Return the location that the macro was defined at.
  ///
  SourceLocation getDefinitionLoc() const { return Location; }

  /// setDefinitionEndLoc - Set the location of the last token in the macro.
  ///
  void setDefinitionEndLoc(SourceLocation EndLoc) { EndLocation = EndLoc; }

  /// getDefinitionEndLoc - Return the location of the last token in the macro.
  ///
  SourceLocation getDefinitionEndLoc() const { return EndLocation; }

  /// \brief Set the location where macro was undefined. Can only be set once.
  void setUndefLoc(SourceLocation UndefLoc) {
    assert(UndefLocation.isInvalid() && "UndefLocation is already set!");
    assert(UndefLoc.isValid() && "Invalid UndefLoc!");
    UndefLocation = UndefLoc;
  }

  /// \brief Get the location where macro was undefined.
  SourceLocation getUndefLoc() const { return UndefLocation; }

  /// \brief Set previous definition of the macro with the same name.
  void setPreviousDefinition(MacroInfo *PreviousDef) {
    PreviousDefinition = PreviousDef;
  }

  /// \brief Get previous definition of the macro with the same name.
  MacroInfo *getPreviousDefinition() { return PreviousDefinition; }

  /// \brief Find macro definition active in the specified source location. If
  /// this macro was not defined there, return NULL.
  const MacroInfo *findDefinitionAtLoc(SourceLocation L,
                                       SourceManager &SM) const;

  /// \brief Get length in characters of the macro definition.
  unsigned getDefinitionLength(SourceManager &SM) const {
    if (IsDefinitionLengthCached)
      return DefinitionLength;
    return getDefinitionLengthSlow(SM);
  }

  /// isIdenticalTo - Return true if the specified macro definition is equal to
  /// this macro in spelling, arguments, and whitespace.  This is used to emit
  /// duplicate definition warnings.  This implements the rules in C99 6.10.3.
  bool isIdenticalTo(const MacroInfo &Other, Preprocessor &PP) const;

  /// setIsBuiltinMacro - Set or clear the isBuiltinMacro flag.
  ///
  void setIsBuiltinMacro(bool Val = true) {
    IsBuiltinMacro = Val;
  }

  /// setIsUsed - Set the value of the IsUsed flag.
  ///
  void setIsUsed(bool Val) {
    IsUsed = Val;
  }

  /// setIsAllowRedefinitionsWithoutWarning - Set the value of the 
  /// IsAllowRedefinitionsWithoutWarning flag.
  void setIsAllowRedefinitionsWithoutWarning(bool Val) {
    IsAllowRedefinitionsWithoutWarning = Val;
  }

  /// \brief Set the value of the IsWarnIfUnused flag.
  void setIsWarnIfUnused(bool val) {
    IsWarnIfUnused = val;
  }

  /// setArgumentList - Set the specified list of identifiers as the argument
  /// list for this macro.
  void setArgumentList(IdentifierInfo* const *List, unsigned NumArgs,
                       llvm::BumpPtrAllocator &PPAllocator) {
    assert(ArgumentList == 0 && NumArguments == 0 &&
           "Argument list already set!");
    if (NumArgs == 0) return;

    NumArguments = NumArgs;
    ArgumentList = PPAllocator.Allocate<IdentifierInfo*>(NumArgs);
    for (unsigned i = 0; i != NumArgs; ++i)
      ArgumentList[i] = List[i];
  }

  /// Arguments - The list of arguments for a function-like macro.  This can be
  /// empty, for, e.g. "#define X()".
  typedef IdentifierInfo* const *arg_iterator;
  bool arg_empty() const { return NumArguments == 0; }
  arg_iterator arg_begin() const { return ArgumentList; }
  arg_iterator arg_end() const { return ArgumentList+NumArguments; }
  unsigned getNumArgs() const { return NumArguments; }

  /// getArgumentNum - Return the argument number of the specified identifier,
  /// or -1 if the identifier is not a formal argument identifier.
  int getArgumentNum(IdentifierInfo *Arg) const {
    for (arg_iterator I = arg_begin(), E = arg_end(); I != E; ++I)
      if (*I == Arg) return I-arg_begin();
    return -1;
  }

  /// Function/Object-likeness.  Keep track of whether this macro has formal
  /// parameters.
  void setIsFunctionLike() { IsFunctionLike = true; }
  bool isFunctionLike() const { return IsFunctionLike; }
  bool isObjectLike() const { return !IsFunctionLike; }

  /// Varargs querying methods.  This can only be set for function-like macros.
  void setIsC99Varargs() { IsC99Varargs = true; }
  void setIsGNUVarargs() { IsGNUVarargs = true; }
  bool isC99Varargs() const { return IsC99Varargs; }
  bool isGNUVarargs() const { return IsGNUVarargs; }
  bool isVariadic() const { return IsC99Varargs | IsGNUVarargs; }

  /// isBuiltinMacro - Return true if this macro is a builtin macro, such as
  /// __LINE__, which requires processing before expansion.
  bool isBuiltinMacro() const { return IsBuiltinMacro; }

  /// isFromAST - Return true if this macro was loaded from an AST file.
  bool isFromAST() const { return IsFromAST; }

  /// setIsFromAST - Set whether this macro was loaded from an AST file.
  void setIsFromAST(bool FromAST = true) { IsFromAST = FromAST; }

  /// \brief Determine whether this macro has changed since it was loaded from
  /// an AST file.
  bool hasChangedAfterLoad() const { return ChangedAfterLoad; }
  
  /// \brief Note whether this macro has changed after it was loaded from an
  /// AST file.
  void setChangedAfterLoad(bool CAL = true) { ChangedAfterLoad = CAL; }
  
  /// isUsed - Return false if this macro is defined in the main file and has
  /// not yet been used.
  bool isUsed() const { return IsUsed; }

  /// isAllowRedefinitionsWithoutWarning - Return true if this macro can be
  /// redefined without warning.
  bool isAllowRedefinitionsWithoutWarning() const {
    return IsAllowRedefinitionsWithoutWarning;
  }

  /// \brief Return true if we should emit a warning if the macro is unused.
  bool isWarnIfUnused() const {
    return IsWarnIfUnused;
  }

  /// getNumTokens - Return the number of tokens that this macro expands to.
  ///
  unsigned getNumTokens() const {
    return ReplacementTokens.size();
  }

  const Token &getReplacementToken(unsigned Tok) const {
    assert(Tok < ReplacementTokens.size() && "Invalid token #");
    return ReplacementTokens[Tok];
  }

  typedef SmallVector<Token, 8>::const_iterator tokens_iterator;
  tokens_iterator tokens_begin() const { return ReplacementTokens.begin(); }
  tokens_iterator tokens_end() const { return ReplacementTokens.end(); }
  bool tokens_empty() const { return ReplacementTokens.empty(); }

  /// AddTokenToBody - Add the specified token to the replacement text for the
  /// macro.
  void AddTokenToBody(const Token &Tok) {
    assert(!IsDefinitionLengthCached &&
          "Changing replacement tokens after definition length got calculated");
    ReplacementTokens.push_back(Tok);
  }

  /// isEnabled - Return true if this macro is enabled: in other words, that we
  /// are not currently in an expansion of this macro.
  bool isEnabled() const { return !IsDisabled; }

  void EnableMacro() {
    assert(IsDisabled && "Cannot enable an already-enabled macro!");
    IsDisabled = false;
  }

  void DisableMacro() {
    assert(!IsDisabled && "Cannot disable an already-disabled macro!");
    IsDisabled = true;
  }

  /// \brief Set the export location for this macro.
  void setVisibility(bool Public, SourceLocation Loc) {
    VisibilityLocation = Loc;
    IsPublic = Public;
  }

  /// \brief Determine whether this macro is part of the public API of its
  /// module.
  bool isPublic() const { return IsPublic; }
  
  /// \brief Determine the location where this macro was explicitly made
  /// public or private within its module.
  SourceLocation getVisibilityLocation() { return VisibilityLocation; }

  /// \brief Determine whether this macro is currently defined (and has not
  /// been #undef'd) or has been hidden.
  bool isDefined() const { return UndefLocation.isInvalid() && !IsHidden; }

  /// \brief Determine whether this macro definition is hidden.
  bool isHidden() const { return IsHidden; }

  /// \brief Set whether this macro definition is hidden.
  void setHidden(bool Val) { IsHidden = Val; }

  /// \brief Determine whether this macro definition is ambiguous with
  /// other macro definitions.
  bool isAmbiguous() const { return IsAmbiguous; }

  /// \brief Set whether this macro definition is ambiguous.
  void setAmbiguous(bool Val) { IsAmbiguous = Val; }
  
private:
  unsigned getDefinitionLengthSlow(SourceManager &SM) const;
};

}  // end namespace clang

#endif
