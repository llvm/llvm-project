//===- ASTDeserializationListener.h - Decl/Type PCH Read Events -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTDeserializationListener class, which is notified
//  by the ASTReader whenever a type or declaration is deserialized.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_AST_DESERIALIZATION_LISTENER_H
#define LLVM_CLANG_FRONTEND_AST_DESERIALIZATION_LISTENER_H

#include "clang/Serialization/ASTBitCodes.h"

namespace clang {

class Decl;
class ASTReader;
class QualType;
class MacroDefinition;
class MacroInfo;
class Module;
  
class ASTDeserializationListener {
protected:
  virtual ~ASTDeserializationListener();

public:

  /// \brief The ASTReader was initialized.
  virtual void ReaderInitialized(ASTReader *Reader) { }

  /// \brief An identifier was deserialized from the AST file.
  virtual void IdentifierRead(serialization::IdentID ID,
                              IdentifierInfo *II) { }
  /// \brief A macro was read from the AST file.
  virtual void MacroRead(serialization::MacroID ID, MacroInfo *MI) { }
  /// \brief A type was deserialized from the AST file. The ID here has the
  ///        qualifier bits already removed, and T is guaranteed to be locally
  ///        unqualified.
  virtual void TypeRead(serialization::TypeIdx Idx, QualType T) { }
  /// \brief A decl was deserialized from the AST file.
  virtual void DeclRead(serialization::DeclID ID, const Decl *D) { }
  /// \brief A selector was read from the AST file.
  virtual void SelectorRead(serialization::SelectorID iD, Selector Sel) { }
  /// \brief A macro definition was read from the AST file.
  virtual void MacroDefinitionRead(serialization::PreprocessedEntityID, 
                                   MacroDefinition *MD) { }
  /// \brief A module definition was read from the AST file.
  virtual void ModuleRead(serialization::SubmoduleID ID, Module *Mod) { }
};

}

#endif
