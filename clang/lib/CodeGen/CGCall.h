//===----- CGCall.h - Encapsulate calling convention details ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGCALL_H
#define CLANG_CODEGEN_CGCALL_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Value.h"
#include "clang/AST/Type.h"
#include "clang/AST/CanonicalType.h"

#include "CGValue.h"

// FIXME: Restructure so we don't have to expose so much stuff.
#include "ABIInfo.h"

namespace llvm {
  struct AttributeWithIndex;
  class Function;
  class Type;
  class Value;

  template<typename T, unsigned> class SmallVector;
}

namespace clang {
  class ASTContext;
  class Decl;
  class FunctionDecl;
  class ObjCMethodDecl;
  class VarDecl;

namespace CodeGen {
  typedef SmallVector<llvm::AttributeWithIndex, 8> AttributeListType;

  struct CallArg {
    RValue RV;
    QualType Ty;
    bool NeedsCopy;
    CallArg(RValue rv, QualType ty, bool needscopy)
    : RV(rv), Ty(ty), NeedsCopy(needscopy)
    { }
  };

  /// CallArgList - Type for representing both the value and type of
  /// arguments in a call.
  class CallArgList :
    public SmallVector<CallArg, 16> {
  public:
    struct Writeback {
      /// The original argument.
      llvm::Value *Address;

      /// The pointee type of the original argument.
      QualType AddressType;

      /// The temporary alloca.
      llvm::Value *Temporary;
    };

    void add(RValue rvalue, QualType type, bool needscopy = false) {
      push_back(CallArg(rvalue, type, needscopy));
    }

    void addFrom(const CallArgList &other) {
      insert(end(), other.begin(), other.end());
      Writebacks.insert(Writebacks.end(),
                        other.Writebacks.begin(), other.Writebacks.end());
    }

    void addWriteback(llvm::Value *address, QualType addressType,
                      llvm::Value *temporary) {
      Writeback writeback;
      writeback.Address = address;
      writeback.AddressType = addressType;
      writeback.Temporary = temporary;
      Writebacks.push_back(writeback);
    }

    bool hasWritebacks() const { return !Writebacks.empty(); }

    typedef SmallVectorImpl<Writeback>::const_iterator writeback_iterator;
    writeback_iterator writeback_begin() const { return Writebacks.begin(); }
    writeback_iterator writeback_end() const { return Writebacks.end(); }

  private:
    SmallVector<Writeback, 1> Writebacks;
  };

  /// A class for recording the number of arguments that a function
  /// signature requires.
  class RequiredArgs {
    /// The number of required arguments, or ~0 if the signature does
    /// not permit optional arguments.
    unsigned NumRequired;
  public:
    enum All_t { All };

    RequiredArgs(All_t _) : NumRequired(~0U) {}
    explicit RequiredArgs(unsigned n) : NumRequired(n) {
      assert(n != ~0U);
    }

    /// Compute the arguments required by the given formal prototype,
    /// given that there may be some additional, non-formal arguments
    /// in play.
    static RequiredArgs forPrototypePlus(const FunctionProtoType *prototype,
                                         unsigned additional) {
      if (!prototype->isVariadic()) return All;
      return RequiredArgs(prototype->getNumArgs() + additional);
    }

    static RequiredArgs forPrototype(const FunctionProtoType *prototype) {
      return forPrototypePlus(prototype, 0);
    }

    static RequiredArgs forPrototype(CanQual<FunctionProtoType> prototype) {
      return forPrototype(prototype.getTypePtr());
    }

    static RequiredArgs forPrototypePlus(CanQual<FunctionProtoType> prototype,
                                         unsigned additional) {
      return forPrototypePlus(prototype.getTypePtr(), additional);
    }

    bool allowsOptionalArgs() const { return NumRequired != ~0U; }
    bool getNumRequiredArgs() const {
      assert(allowsOptionalArgs());
      return NumRequired;
    }

    unsigned getOpaqueData() const { return NumRequired; }
    static RequiredArgs getFromOpaqueData(unsigned value) {
      if (value == ~0U) return All;
      return RequiredArgs(value);
    }
  };

  /// FunctionArgList - Type for representing both the decl and type
  /// of parameters to a function. The decl must be either a
  /// ParmVarDecl or ImplicitParamDecl.
  class FunctionArgList : public SmallVector<const VarDecl*, 16> {
  };

  /// CGFunctionInfo - Class to encapsulate the information about a
  /// function definition.
  class CGFunctionInfo : public llvm::FoldingSetNode {
    struct ArgInfo {
      CanQualType type;
      ABIArgInfo info;
    };

    /// The LLVM::CallingConv to use for this function (as specified by the
    /// user).
    unsigned CallingConvention : 8;

    /// The LLVM::CallingConv to actually use for this function, which may
    /// depend on the ABI.
    unsigned EffectiveCallingConvention : 8;

    /// The clang::CallingConv that this was originally created with.
    unsigned ASTCallingConvention : 8;

    /// Whether this function is noreturn.
    unsigned NoReturn : 1;

    /// Whether this function is returns-retained.
    unsigned ReturnsRetained : 1;

    /// How many arguments to pass inreg.
    unsigned HasRegParm : 1;
    unsigned RegParm : 4;

    RequiredArgs Required;

    unsigned NumArgs;
    ArgInfo *getArgsBuffer() {
      return reinterpret_cast<ArgInfo*>(this+1);
    }
    const ArgInfo *getArgsBuffer() const {
      return reinterpret_cast<const ArgInfo*>(this + 1);
    }

    CGFunctionInfo() : Required(RequiredArgs::All) {}

  public:
    static CGFunctionInfo *create(unsigned llvmCC,
                                  const FunctionType::ExtInfo &extInfo,
                                  CanQualType resultType,
                                  ArrayRef<CanQualType> argTypes,
                                  RequiredArgs required);

    typedef const ArgInfo *const_arg_iterator;
    typedef ArgInfo *arg_iterator;

    const_arg_iterator arg_begin() const { return getArgsBuffer() + 1; }
    const_arg_iterator arg_end() const { return getArgsBuffer() + 1 + NumArgs; }
    arg_iterator arg_begin() { return getArgsBuffer() + 1; }
    arg_iterator arg_end() { return getArgsBuffer() + 1 + NumArgs; }

    unsigned  arg_size() const { return NumArgs; }

    bool isVariadic() const { return Required.allowsOptionalArgs(); }
    RequiredArgs getRequiredArgs() const { return Required; }

    bool isNoReturn() const { return NoReturn; }

    /// In ARC, whether this function retains its return value.  This
    /// is not always reliable for call sites.
    bool isReturnsRetained() const { return ReturnsRetained; }

    /// getASTCallingConvention() - Return the AST-specified calling
    /// convention.
    CallingConv getASTCallingConvention() const {
      return CallingConv(ASTCallingConvention);
    }

    /// getCallingConvention - Return the user specified calling
    /// convention, which has been translated into an LLVM CC.
    unsigned getCallingConvention() const { return CallingConvention; }

    /// getEffectiveCallingConvention - Return the actual calling convention to
    /// use, which may depend on the ABI.
    unsigned getEffectiveCallingConvention() const {
      return EffectiveCallingConvention;
    }
    void setEffectiveCallingConvention(unsigned Value) {
      EffectiveCallingConvention = Value;
    }

    bool getHasRegParm() const { return HasRegParm; }
    unsigned getRegParm() const { return RegParm; }

    FunctionType::ExtInfo getExtInfo() const {
      return FunctionType::ExtInfo(isNoReturn(),
                                   getHasRegParm(), getRegParm(),
                                   getASTCallingConvention(),
                                   isReturnsRetained());
    }

    CanQualType getReturnType() const { return getArgsBuffer()[0].type; }

    ABIArgInfo &getReturnInfo() { return getArgsBuffer()[0].info; }
    const ABIArgInfo &getReturnInfo() const { return getArgsBuffer()[0].info; }

    void Profile(llvm::FoldingSetNodeID &ID) {
      ID.AddInteger(getASTCallingConvention());
      ID.AddBoolean(NoReturn);
      ID.AddBoolean(ReturnsRetained);
      ID.AddBoolean(HasRegParm);
      ID.AddInteger(RegParm);
      ID.AddInteger(Required.getOpaqueData());
      getReturnType().Profile(ID);
      for (arg_iterator it = arg_begin(), ie = arg_end(); it != ie; ++it)
        it->type.Profile(ID);
    }
    static void Profile(llvm::FoldingSetNodeID &ID,
                        const FunctionType::ExtInfo &info,
                        RequiredArgs required,
                        CanQualType resultType,
                        ArrayRef<CanQualType> argTypes) {
      ID.AddInteger(info.getCC());
      ID.AddBoolean(info.getNoReturn());
      ID.AddBoolean(info.getProducesResult());
      ID.AddBoolean(info.getHasRegParm());
      ID.AddInteger(info.getRegParm());
      ID.AddInteger(required.getOpaqueData());
      resultType.Profile(ID);
      for (ArrayRef<CanQualType>::iterator
             i = argTypes.begin(), e = argTypes.end(); i != e; ++i) {
        i->Profile(ID);
      }
    }
  };
  
  /// ReturnValueSlot - Contains the address where the return value of a 
  /// function can be stored, and whether the address is volatile or not.
  class ReturnValueSlot {
    llvm::PointerIntPair<llvm::Value *, 1, bool> Value;

  public:
    ReturnValueSlot() {}
    ReturnValueSlot(llvm::Value *Value, bool IsVolatile)
      : Value(Value, IsVolatile) {}

    bool isNull() const { return !getValue(); }
    
    bool isVolatile() const { return Value.getInt(); }
    llvm::Value *getValue() const { return Value.getPointer(); }
  };
  
}  // end namespace CodeGen
}  // end namespace clang

#endif
