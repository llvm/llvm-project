//===----- ConstantBuilder.h - Builder for LLVM IR constants ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class provides a convenient interface for building complex
// global initializers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CONSTANTBUILDER_H
#define LLVM_CLANG_LIB_CODEGEN_CONSTANTBUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"

#include "CodeGenModule.h"

namespace clang {
namespace CodeGen {

class ConstantBuilder;

/// A convenience builder class for complex constant initializers,
/// especially for anonymous global structures used by various language
/// runtimes.
///
/// The basic usage pattern is expected to be something like:
///    ConstantBuilder builder(CGM);
///    auto toplevel = builder.beginStruct();
///    toplevel.addInt(CGM.SizeTy, widgets.size());
///    auto widgetArray = builder.beginArray();
///    for (auto &widget : widgets) {
///      auto widgetDesc = widgetArray.beginStruct();
///      widgetDesc.addInt(CGM.SizeTy, widget.getPower());
///      widgetDesc.add(CGM.GetAddrOfConstantString(widget.getName()));
///      widgetDesc.add(CGM.GetAddrOfGlobal(widget.getInitializerDecl()));
///      widgetArray.add(widgetDesc.finish());
///    }
///    toplevel.add(widgetArray.finish());
///    auto global = toplevel.finishAndCreateGlobal("WIDGET_LIST", Align,
///                                                 /*constant*/ true);
class ConstantBuilder {
  CodeGenModule &CGM;
  llvm::SmallVector<llvm::Constant*, 16> Buffer;
  bool Frozen = false;

public:
  explicit ConstantBuilder(CodeGenModule &CGM) : CGM(CGM) {}

  ~ConstantBuilder() {
    assert(Buffer.empty() && "didn't claim all values out of buffer");
  }

  class ArrayBuilder;
  class StructBuilder;

  class AggregateBuilder {
  protected:
    ConstantBuilder &Builder;
    AggregateBuilder *Parent;
    size_t Begin;
    bool Finished = false;
    bool Frozen = false;

    AggregateBuilder(ConstantBuilder &builder, AggregateBuilder *parent)
        : Builder(builder), Parent(parent), Begin(builder.Buffer.size()) {
      if (parent) {
        assert(!parent->Frozen && "parent already has child builder active");
        parent->Frozen = true;
      } else {
        assert(!builder.Frozen && "builder already has child builder active");
        builder.Frozen = true;
      }
    }

    ~AggregateBuilder() {
      assert(Finished && "didn't claim value from aggregate builder");
    }

    void markFinished() {
      assert(!Frozen && "child builder still active");
      assert(!Finished && "builder already finished");
      Finished = true;
      if (Parent) {
        assert(Parent->Frozen &&
               "parent not frozen while child builder active");
        Parent->Frozen = false;
      } else {
        assert(Builder.Frozen &&
               "builder not frozen while child builder active");
        Builder.Frozen = false;
      }
    }

  public:
    // Not copyable.
    AggregateBuilder(const AggregateBuilder &) = delete;
    AggregateBuilder &operator=(const AggregateBuilder &) = delete;

    // Movable, mostly to allow returning.  But we have to write this out
    // properly to satisfy the assert in the destructor.
    AggregateBuilder(AggregateBuilder &&other)
      : Builder(other.Builder), Parent(other.Parent), Begin(other.Begin),
        Finished(other.Finished), Frozen(other.Frozen) {
      other.Finished = false;
    }
    AggregateBuilder &operator=(AggregateBuilder &&other) = delete;

    void add(llvm::Constant *value) {
      assert(!Finished && "cannot add more values after finishing builder");
      Builder.Buffer.push_back(value);
    }

    void addSize(CharUnits size) {
      add(Builder.CGM.getSize(size));
    }

    void addInt(llvm::IntegerType *intTy, uint64_t value,
                bool isSigned = false) {
      add(llvm::ConstantInt::get(intTy, value, isSigned));
    }

    void addNullPointer(llvm::PointerType *ptrTy) {
      add(llvm::ConstantPointerNull::get(ptrTy));
    }

    ArrayRef<llvm::Constant*> getGEPIndicesToCurrentPosition(
                             llvm::SmallVectorImpl<llvm::Constant*> &indices) {
      getGEPIndicesTo(indices, Builder.Buffer.size());
      return indices;
    }

    ArrayBuilder beginArray(llvm::Type *eltTy = nullptr);
    StructBuilder beginStruct(llvm::StructType *structTy = nullptr);

  private:
    void getGEPIndicesTo(llvm::SmallVectorImpl<llvm::Constant*> &indices,
                         size_t position) const {
      // Recurse on the parent builder if present.
      if (Parent) {
        Parent->getGEPIndicesTo(indices, Begin);

      // Otherwise, add an index to drill into the first level of pointer. 
      } else {
        assert(indices.empty());
        indices.push_back(llvm::ConstantInt::get(Builder.CGM.SizeTy, 0));
      }

      assert(position >= Begin);
      indices.push_back(llvm::ConstantInt::get(Builder.CGM.SizeTy,
                                               position - Begin));
    }
  };

  class ArrayBuilder : public AggregateBuilder {
    llvm::Type *EltTy;
    friend class ConstantBuilder;
    ArrayBuilder(ConstantBuilder &builder, AggregateBuilder *parent,
                 llvm::Type *eltTy)
      : AggregateBuilder(builder, parent), EltTy(eltTy) {}
  public:
    size_t size() const {
      assert(!Finished);
      assert(!Frozen);
      assert(Begin <= Builder.Buffer.size());
      return Builder.Buffer.size() - Begin;
    }

    /// Form an array constant from the values that have been added to this
    /// builder.
    llvm::Constant *finish() {
      markFinished();

      auto &buffer = Builder.Buffer;
      assert((Begin < buffer.size() ||
              (Begin == buffer.size() && EltTy))
             && "didn't add any array elements without element type");
      auto elts = llvm::makeArrayRef(buffer).slice(Begin);
      auto eltTy = EltTy ? EltTy : elts[0]->getType();
      auto type = llvm::ArrayType::get(eltTy, elts.size());
      auto constant = llvm::ConstantArray::get(type, elts);
      buffer.erase(buffer.begin() + Begin, buffer.end());
      return constant;
    }

    template <class... As>
    llvm::GlobalVariable *finishAndCreateGlobal(As &&...args) {
      assert(!Parent && "finishing non-root builder");
      return Builder.createGlobal(finish(), std::forward<As>(args)...);
    }
  };

  ArrayBuilder beginArray(llvm::Type *eltTy = nullptr) {
    return ArrayBuilder(*this, nullptr, eltTy);
  }

  class StructBuilder : public AggregateBuilder {
    llvm::StructType *Ty;
    friend class ConstantBuilder;
    StructBuilder(ConstantBuilder &builder, AggregateBuilder *parent,
                  llvm::StructType *ty)
      : AggregateBuilder(builder, parent), Ty(ty) {}
  public:
    /// Finish the struct.
    llvm::Constant *finish(bool packed = false) {
      markFinished();

      auto &buffer = Builder.Buffer;
      assert(Begin < buffer.size() && "didn't add any struct elements?");
      auto elts = llvm::makeArrayRef(buffer).slice(Begin);

      llvm::Constant *constant;
      if (Ty) {
        constant = llvm::ConstantStruct::get(Ty, elts);
      } else {
        constant = llvm::ConstantStruct::getAnon(elts, packed);
      }

      buffer.erase(buffer.begin() + Begin, buffer.end());
      return constant;
    }

    template <class... As>
    llvm::GlobalVariable *finishAndCreateGlobal(As &&...args) {
      assert(!Parent && "finishing non-root builder");
      return Builder.createGlobal(finish(), std::forward<As>(args)...);
    }
  };

  StructBuilder beginStruct(llvm::StructType *structTy = nullptr) {
    return StructBuilder(*this, nullptr, structTy);
  }

  llvm::GlobalVariable *createGlobal(llvm::Constant *initializer,
                                     StringRef name,
                                     CharUnits alignment,
                                     bool constant = false,
                                     llvm::GlobalValue::LinkageTypes linkage
                                       = llvm::GlobalValue::InternalLinkage,
                                     unsigned addressSpace = 0) {
    auto GV = new llvm::GlobalVariable(CGM.getModule(),
                                       initializer->getType(),
                                       constant,
                                       linkage,
                                       initializer,
                                       name,
                                       /*insert before*/ nullptr,
                                       llvm::GlobalValue::NotThreadLocal,
                                       addressSpace);
    GV->setAlignment(alignment.getQuantity());
    return GV;
  }
};

inline ConstantBuilder::ArrayBuilder
ConstantBuilder::AggregateBuilder::beginArray(llvm::Type *eltTy) {
  return ArrayBuilder(Builder, this, eltTy);
}

inline ConstantBuilder::StructBuilder
ConstantBuilder::AggregateBuilder::beginStruct(llvm::StructType *structTy) {
  return StructBuilder(Builder, this, structTy);
}

}  // end namespace CodeGen
}  // end namespace clang

#endif
