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

#include "Descriptor.h"
#include "Source.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace interp {
class Program;
class ByteCodeEmitter;
class Pointer;
enum PrimType : uint8_t;

/// Describes a scope block.
///
/// The block gathers all the descriptors of the locals defined in this block.
class Scope final {
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

  llvm::iterator_range<LocalVectorTy::const_iterator> locals() const {
    return llvm::make_range(Descriptors.begin(), Descriptors.end());
  }

  llvm::iterator_range<LocalVectorTy::const_reverse_iterator>
  locals_reverse() const {
    return llvm::reverse(Descriptors);
  }

private:
  /// Object descriptors in this block.
  LocalVectorTy Descriptors;
};

using FunctionDeclTy =
    llvm::PointerUnion<const FunctionDecl *, const BlockExpr *>;

/// Bytecode function.
///
/// Contains links to the bytecode of the function, as well as metadata
/// describing all arguments and stack-local variables.
///
/// # Calling Convention
///
/// When calling a function, all argument values must be on the stack.
///
/// If the function has a This pointer (i.e. hasThisPointer() returns true,
/// the argument values need to be preceeded by a Pointer for the This object.
///
/// If the function uses Return Value Optimization, the arguments (and
/// potentially the This pointer) need to be preceeded by a Pointer pointing
/// to the location to construct the returned value.
///
/// After the function has been called, it will remove all arguments,
/// including RVO and This pointer, from the stack.
///
class Function final {
public:
  enum class FunctionKind {
    Normal,
    Ctor,
    Dtor,
    LambdaStaticInvoker,
    LambdaCallOperator,
    CopyOrMoveOperator,
  };
  using ParamDescriptor = std::pair<PrimType, Descriptor *>;

  /// Returns the size of the function's local stack.
  unsigned getFrameSize() const { return FrameSize; }
  /// Returns the size of the argument stack.
  unsigned getArgSize() const { return ArgSize; }

  /// Returns a pointer to the start of the code.
  CodePtr getCodeBegin() const { return Code.data(); }
  /// Returns a pointer to the end of the code.
  CodePtr getCodeEnd() const { return Code.data() + Code.size(); }

  /// Returns the original FunctionDecl.
  const FunctionDecl *getDecl() const {
    return dyn_cast<const FunctionDecl *>(Source);
  }
  const BlockExpr *getExpr() const {
    return dyn_cast<const BlockExpr *>(Source);
  }

  /// Returns the name of the function decl this code
  /// was generated for.
  std::string getName() const {
    if (!Source || !getDecl())
      return "<<expr>>";

    return getDecl()->getQualifiedNameAsString();
  }

  /// Returns a parameter descriptor.
  ParamDescriptor getParamDescriptor(unsigned Offset) const;

  /// Checks if the first argument is a RVO pointer.
  bool hasRVO() const { return HasRVO; }

  bool hasNonNullAttr() const { return getDecl()->hasAttr<NonNullAttr>(); }

  /// Range over the scope blocks.
  llvm::iterator_range<llvm::SmallVector<Scope, 2>::const_iterator>
  scopes() const {
    return llvm::make_range(Scopes.begin(), Scopes.end());
  }

  /// Range over argument types.
  using arg_reverse_iterator =
      SmallVectorImpl<PrimType>::const_reverse_iterator;
  llvm::iterator_range<arg_reverse_iterator> args_reverse() const {
    return llvm::reverse(ParamTypes);
  }

  /// Returns a specific scope.
  Scope &getScope(unsigned Idx) { return Scopes[Idx]; }
  const Scope &getScope(unsigned Idx) const { return Scopes[Idx]; }

  /// Returns the source information at a given PC.
  SourceInfo getSource(CodePtr PC) const;

  /// Checks if the function is valid to call.
  bool isValid() const { return IsValid || isLambdaStaticInvoker(); }

  /// Checks if the function is virtual.
  bool isVirtual() const { return Virtual; };
  bool isImmediate() const { return Immediate; }
  bool isConstexpr() const { return Constexpr; }

  /// Checks if the function is a constructor.
  bool isConstructor() const { return Kind == FunctionKind::Ctor; }
  /// Checks if the function is a destructor.
  bool isDestructor() const { return Kind == FunctionKind::Dtor; }
  /// Checks if the function is copy or move operator.
  bool isCopyOrMoveOperator() const {
    return Kind == FunctionKind::CopyOrMoveOperator;
  }

  /// Returns whether this function is a lambda static invoker,
  /// which we generate custom byte code for.
  bool isLambdaStaticInvoker() const {
    return Kind == FunctionKind::LambdaStaticInvoker;
  }

  /// Returns whether this function is the call operator
  /// of a lambda record decl.
  bool isLambdaCallOperator() const {
    return Kind == FunctionKind::LambdaCallOperator;
  }

  /// Returns the parent record decl, if any.
  const CXXRecordDecl *getParentDecl() const {
    if (const auto *MD = dyn_cast_if_present<CXXMethodDecl>(
            dyn_cast<const FunctionDecl *>(Source)))
      return MD->getParent();
    return nullptr;
  }

  /// Checks if the function is fully done compiling.
  bool isFullyCompiled() const { return IsFullyCompiled; }

  bool hasThisPointer() const { return HasThisPointer; }

  /// Checks if the function already has a body attached.
  bool hasBody() const { return HasBody; }

  /// Checks if the function is defined.
  bool isDefined() const { return Defined; }

  bool isVariadic() const { return Variadic; }

  unsigned getNumParams() const { return ParamTypes.size(); }

  /// Returns the number of parameter this function takes when it's called,
  /// i.e excluding the instance pointer and the RVO pointer.
  unsigned getNumWrittenParams() const {
    assert(getNumParams() >= (unsigned)(hasThisPointer() + hasRVO()));
    return getNumParams() - hasThisPointer() - hasRVO();
  }
  unsigned getWrittenArgSize() const {
    return ArgSize - (align(primSize(PT_Ptr)) * (hasThisPointer() + hasRVO()));
  }

  bool isThisPointerExplicit() const {
    if (const auto *MD = dyn_cast_if_present<CXXMethodDecl>(
            dyn_cast<const FunctionDecl *>(Source)))
      return MD->isExplicitObjectMemberFunction();
    return false;
  }

  unsigned getParamOffset(unsigned ParamIndex) const {
    return ParamOffsets[ParamIndex];
  }

  PrimType getParamType(unsigned ParamIndex) const {
    return ParamTypes[ParamIndex];
  }

private:
  /// Construct a function representing an actual function.
  Function(Program &P, FunctionDeclTy Source, unsigned ArgSize,
           llvm::SmallVectorImpl<PrimType> &&ParamTypes,
           llvm::DenseMap<unsigned, ParamDescriptor> &&Params,
           llvm::SmallVectorImpl<unsigned> &&ParamOffsets, bool HasThisPointer,
           bool HasRVO, bool IsLambdaStaticInvoker);

  /// Sets the code of a function.
  void setCode(FunctionDeclTy Source, unsigned NewFrameSize,
               llvm::SmallVector<std::byte> &&NewCode, SourceMap &&NewSrcMap,
               llvm::SmallVector<Scope, 2> &&NewScopes, bool NewHasBody) {
    this->Source = Source;
    FrameSize = NewFrameSize;
    Code = std::move(NewCode);
    SrcMap = std::move(NewSrcMap);
    Scopes = std::move(NewScopes);
    IsValid = true;
    HasBody = NewHasBody;
  }

  void setIsFullyCompiled(bool FC) { IsFullyCompiled = FC; }
  void setDefined(bool D) { Defined = D; }

private:
  friend class Program;
  friend class ByteCodeEmitter;
  friend class Context;

  /// Program reference.
  Program &P;
  /// Function Kind.
  FunctionKind Kind;
  /// Declaration this function was compiled from.
  FunctionDeclTy Source;
  /// Local area size: storage + metadata.
  unsigned FrameSize = 0;
  /// Size of the argument stack.
  unsigned ArgSize;
  /// Program code.
  llvm::SmallVector<std::byte> Code;
  /// Opcode-to-expression mapping.
  SourceMap SrcMap;
  /// List of block descriptors.
  llvm::SmallVector<Scope, 2> Scopes;
  /// List of argument types.
  llvm::SmallVector<PrimType, 8> ParamTypes;
  /// Map from byte offset to parameter descriptor.
  llvm::DenseMap<unsigned, ParamDescriptor> Params;
  /// List of parameter offsets.
  llvm::SmallVector<unsigned, 8> ParamOffsets;
  /// Flag to indicate if the function is valid.
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsValid : 1;
  /// Flag to indicate if the function is done being
  /// compiled to bytecode.
  LLVM_PREFERRED_TYPE(bool)
  unsigned IsFullyCompiled : 1;
  /// Flag indicating if this function takes the this pointer
  /// as the first implicit argument
  LLVM_PREFERRED_TYPE(bool)
  unsigned HasThisPointer : 1;
  /// Whether this function has Return Value Optimization, i.e.
  /// the return value is constructed in the caller's stack frame.
  /// This is done for functions that return non-primive values.
  LLVM_PREFERRED_TYPE(bool)
  unsigned HasRVO : 1;
  /// If we've already compiled the function's body.
  LLVM_PREFERRED_TYPE(bool)
  unsigned HasBody : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Defined : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Variadic : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Virtual : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Immediate : 1;
  LLVM_PREFERRED_TYPE(bool)
  unsigned Constexpr : 1;

public:
  /// Dumps the disassembled bytecode to \c llvm::errs().
  void dump() const;
  void dump(llvm::raw_ostream &OS) const;
};

} // namespace interp
} // namespace clang

#endif
