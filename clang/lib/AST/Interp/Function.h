//===--- Function.h - Bytecode function for the VM --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the Function class which holds all bytecode function-specific data.
//
// The scope class which describes local variables is also defined here.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_INTERP_FUNCTION_H
#define LLVM_CLANG_AST_INTERP_FUNCTION_H

#include "Pointer.h"
#include "Source.h"
#include "clang/AST/Decl.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace interp {
class Program;
class ByteCodeEmitter;
enum PrimType : uint32_t;

/// Describes a scope block.
///
/// The block gathers all the descriptors of the locals defined in this block.
class Scope {
public:
  /// Information about a local's storage.
  struct Local {
    /// Offset of the local in frame.
    unsigned Offset;
    /// Descriptor of the local.
    Descriptor *Desc;
  };

  using LocalVectorTy = llvm::SmallVector<Local, 8>;

  Scope(LocalVectorTy &&Descriptors) : Descriptors(std::move(Descriptors)) {}

  llvm::iterator_range<LocalVectorTy::iterator> locals() {
    return llvm::make_range(Descriptors.begin(), Descriptors.end());
  }

private:
  /// Object descriptors in this block.
  LocalVectorTy Descriptors;
};

/// Bytecode function.
///
/// Contains links to the bytecode of the function, as well as metadata
/// describing all arguments and stack-local variables.
class Function {
public:
  using ParamDescriptor = std::pair<PrimType, Descriptor *>;

  /// Returns the size of the function's local stack.
  unsigned getFrameSize() const { return FrameSize; }
  /// Returns the size of the argument stackx
  unsigned getArgSize() const { return ArgSize; }

  /// Returns a pointer to the start of the code.
  CodePtr getCodeBegin() const;
  /// Returns a pointer to the end of the code.
  CodePtr getCodeEnd() const;

  /// Returns the original FunctionDecl.
  const FunctionDecl *getDecl() const { return F; }

  /// Returns the location.
  SourceLocation getLoc() const { return Loc; }

  /// Returns a parameter descriptor.
  ParamDescriptor getParamDescriptor(unsigned Offset) const;

  /// Checks if the first argument is a RVO pointer.
  bool hasRVO() const { return ParamTypes.size() != Params.size(); }

  /// Range over the scope blocks.
  llvm::iterator_range<llvm::SmallVector<Scope, 2>::iterator> scopes() {
    return llvm::make_range(Scopes.begin(), Scopes.end());
  }

  /// Range over argument types.
  using arg_reverse_iterator = SmallVectorImpl<PrimType>::reverse_iterator;
  llvm::iterator_range<arg_reverse_iterator> args_reverse() {
    return llvm::make_range(ParamTypes.rbegin(), ParamTypes.rend());
  }

  /// Returns a specific scope.
  Scope &getScope(unsigned Idx) { return Scopes[Idx]; }

  /// Returns the source information at a given PC.
  SourceInfo getSource(CodePtr PC) const;

  /// Checks if the function is valid to call in constexpr.
  bool isConstexpr() const { return IsValid; }

  /// Checks if the function is virtual.
  bool isVirtual() const;

  /// Checks if the function is a constructor.
  bool isConstructor() const { return isa<CXXConstructorDecl>(F); }

private:
  /// Construct a function representing an actual function.
  Function(Program &P, const FunctionDecl *F, unsigned ArgSize,
           llvm::SmallVector<PrimType, 8> &&ParamTypes,
           llvm::DenseMap<unsigned, ParamDescriptor> &&Params);

  /// Sets the code of a function.
  void setCode(unsigned NewFrameSize, std::vector<char> &&NewCode, SourceMap &&NewSrcMap,
               llvm::SmallVector<Scope, 2> &&NewScopes) {
    FrameSize = NewFrameSize;
    Code = std::move(NewCode);
    SrcMap = std::move(NewSrcMap);
    Scopes = std::move(NewScopes);
    IsValid = true;
  }

private:
  friend class Program;
  friend class ByteCodeEmitter;

  /// Program reference.
  Program &P;
  /// Location of the executed code.
  SourceLocation Loc;
  /// Declaration this function was compiled from.
  const FunctionDecl *F;
  /// Local area size: storage + metadata.
  unsigned FrameSize;
  /// Size of the argument stack.
  unsigned ArgSize;
  /// Program code.
  std::vector<char> Code;
  /// Opcode-to-expression mapping.
  SourceMap SrcMap;
  /// List of block descriptors.
  llvm::SmallVector<Scope, 2> Scopes;
  /// List of argument types.
  llvm::SmallVector<PrimType, 8> ParamTypes;
  /// Map from byte offset to parameter descriptor.
  llvm::DenseMap<unsigned, ParamDescriptor> Params;
  /// Flag to indicate if the function is valid.
  bool IsValid = false;

public:
  /// Dumps the disassembled bytecode to \c llvm::errs().
  void dump() const;
  void dump(llvm::raw_ostream &OS) const;
};

} // namespace interp
} // namespace clang

#endif
