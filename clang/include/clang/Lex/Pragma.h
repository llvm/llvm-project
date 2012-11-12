//===--- Pragma.h - Pragma registration and handling ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PragmaHandler and PragmaTable interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PRAGMA_H
#define LLVM_CLANG_PRAGMA_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

namespace clang {
  class Preprocessor;
  class Token;
  class IdentifierInfo;
  class PragmaNamespace;

  /**
   * \brief Describes how the pragma was introduced, e.g., with \#pragma,
   * _Pragma, or __pragma.
   */
  enum PragmaIntroducerKind {
    /**
     * \brief The pragma was introduced via \#pragma.
     */
    PIK_HashPragma,
    
    /**
     * \brief The pragma was introduced via the C99 _Pragma(string-literal).
     */
    PIK__Pragma,
    
    /**
     * \brief The pragma was introduced via the Microsoft 
     * __pragma(token-string).
     */
    PIK___pragma
  };
  
/// PragmaHandler - Instances of this interface defined to handle the various
/// pragmas that the language front-end uses.  Each handler optionally has a
/// name (e.g. "pack") and the HandlePragma method is invoked when a pragma with
/// that identifier is found.  If a handler does not match any of the declared
/// pragmas the handler with a null identifier is invoked, if it exists.
///
/// Note that the PragmaNamespace class can be used to subdivide pragmas, e.g.
/// we treat "\#pragma STDC" and "\#pragma GCC" as namespaces that contain other
/// pragmas.
class PragmaHandler {
  std::string Name;
public:
  explicit PragmaHandler(StringRef name) : Name(name) {}
  PragmaHandler() {}
  virtual ~PragmaHandler();

  StringRef getName() const { return Name; }
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken) = 0;

  /// getIfNamespace - If this is a namespace, return it.  This is equivalent to
  /// using a dynamic_cast, but doesn't require RTTI.
  virtual PragmaNamespace *getIfNamespace() { return 0; }
};

/// EmptyPragmaHandler - A pragma handler which takes no action, which can be
/// used to ignore particular pragmas.
class EmptyPragmaHandler : public PragmaHandler {
public:
  EmptyPragmaHandler();

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

/// PragmaNamespace - This PragmaHandler subdivides the namespace of pragmas,
/// allowing hierarchical pragmas to be defined.  Common examples of namespaces
/// are "\#pragma GCC", "\#pragma STDC", and "\#pragma omp", but any namespaces
/// may be (potentially recursively) defined.
class PragmaNamespace : public PragmaHandler {
  /// Handlers - This is a map of the handlers in this namespace with their name
  /// as key.
  ///
  llvm::StringMap<PragmaHandler*> Handlers;
public:
  explicit PragmaNamespace(StringRef Name) : PragmaHandler(Name) {}
  virtual ~PragmaNamespace();

  /// FindHandler - Check to see if there is already a handler for the
  /// specified name.  If not, return the handler for the null name if it
  /// exists, otherwise return null.  If IgnoreNull is true (the default) then
  /// the null handler isn't returned on failure to match.
  PragmaHandler *FindHandler(StringRef Name,
                             bool IgnoreNull = true) const;

  /// AddPragma - Add a pragma to this namespace.
  ///
  void AddPragma(PragmaHandler *Handler);

  /// RemovePragmaHandler - Remove the given handler from the
  /// namespace.
  void RemovePragmaHandler(PragmaHandler *Handler);

  bool IsEmpty() {
    return Handlers.empty();
  }

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);

  virtual PragmaNamespace *getIfNamespace() { return this; }
};


}  // end namespace clang

#endif
