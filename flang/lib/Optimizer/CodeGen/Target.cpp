//===-- Target.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/CodeGen/Target.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "flang-codegen-target"

using namespace fir;

namespace fir::details {
llvm::StringRef Attributes::getIntExtensionAttrName() const {
  // The attribute names are available via LLVM dialect interfaces
  // like getZExtAttrName(), getByValAttrName(), etc., so we'd better
  // use them than literals.
  if (isZeroExt())
    return "llvm.zeroext";
  else if (isSignExt())
    return "llvm.signext";
  return {};
}
} // namespace fir::details

// Reduce a REAL/float type to the floating point semantics.
static const llvm::fltSemantics &floatToSemantics(const KindMapping &kindMap,
                                                  mlir::Type type) {
  assert(isa_real(type));
  return mlir::cast<mlir::FloatType>(type).getFloatSemantics();
}

static void typeTodo(const llvm::fltSemantics *sem, mlir::Location loc,
                     const std::string &context) {
  if (sem == &llvm::APFloat::IEEEhalf()) {
    TODO(loc, "COMPLEX(KIND=2): for " + context + " type");
  } else if (sem == &llvm::APFloat::BFloat()) {
    TODO(loc, "COMPLEX(KIND=3): " + context + " type");
  } else if (sem == &llvm::APFloat::x87DoubleExtended()) {
    TODO(loc, "COMPLEX(KIND=10): " + context + " type");
  } else {
    TODO(loc, "complex for this precision for " + context + " type");
  }
}

namespace {
template <typename S>
struct GenericTarget : public CodeGenSpecifics {
  using CodeGenSpecifics::CodeGenSpecifics;
  using AT = CodeGenSpecifics::Attributes;

  mlir::Type complexMemoryType(mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    // Use a type that will be translated into LLVM as:
    // { t, t }   struct of 2 eleTy
    return mlir::TupleType::get(eleTy.getContext(),
                                mlir::TypeRange{eleTy, eleTy});
  }

  mlir::Type boxcharMemoryType(mlir::Type eleTy) const override {
    auto idxTy = mlir::IntegerType::get(eleTy.getContext(), S::defaultWidth);
    auto ptrTy = fir::ReferenceType::get(eleTy);
    // Use a type that will be translated into LLVM as:
    // { t*, index }
    return mlir::TupleType::get(eleTy.getContext(),
                                mlir::TypeRange{ptrTy, idxTy});
  }

  Marshalling boxcharArgumentType(mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    auto idxTy = mlir::IntegerType::get(eleTy.getContext(), S::defaultWidth);
    auto ptrTy = fir::ReferenceType::get(eleTy);
    marshal.emplace_back(ptrTy, AT{});
    // Characters are passed in a split format with all pointers first (in the
    // declared position) and all LEN arguments appended after all of the dummy
    // arguments.
    // NB: Other conventions/ABIs can/should be supported via options.
    marshal.emplace_back(idxTy, AT{/*alignment=*/0, /*byval=*/false,
                                   /*sret=*/false, /*append=*/true});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  structArgumentType(mlir::Location loc, fir::RecordType,
                     const Marshalling &) const override {
    TODO(loc, "passing VALUE BIND(C) derived type for this target");
  }

  CodeGenSpecifics::Marshalling
  structReturnType(mlir::Location loc, fir::RecordType ty) const override {
    TODO(loc, "returning BIND(C) derived type for this target");
  }

  CodeGenSpecifics::Marshalling
  integerArgumentType(mlir::Location loc,
                      mlir::IntegerType argTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    AT::IntegerExtension intExt = AT::IntegerExtension::None;
    if (argTy.getWidth() < getCIntTypeWidth()) {
      // isSigned() and isUnsigned() branches below are dead code currently.
      // If needed, we can generate calls with signed/unsigned argument types
      // to more precisely match C side (e.g. for Fortran runtime functions
      // with 'unsigned short' arguments).
      if (argTy.isSigned())
        intExt = AT::IntegerExtension::Sign;
      else if (argTy.isUnsigned())
        intExt = AT::IntegerExtension::Zero;
      else if (argTy.isSignless()) {
        // Zero extend for 'i1' and sign extend for other types.
        if (argTy.getWidth() == 1)
          intExt = AT::IntegerExtension::Zero;
        else
          intExt = AT::IntegerExtension::Sign;
      }
    }

    marshal.emplace_back(argTy, AT{/*alignment=*/0, /*byval=*/false,
                                   /*sret=*/false, /*append=*/false,
                                   /*intExt=*/intExt});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  integerReturnType(mlir::Location loc,
                    mlir::IntegerType argTy) const override {
    return integerArgumentType(loc, argTy);
  }

  // Width of 'int' type is 32-bits for almost all targets, except
  // for AVR and MSP430 (see TargetInfo initializations
  // in clang/lib/Basic/Targets).
  unsigned char getCIntTypeWidth() const override { return 32; }
};
} // namespace

//===----------------------------------------------------------------------===//
// i386 (x86 32 bit) linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetI386 : public GenericTarget<TargetI386> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 32;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location, mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    CodeGenSpecifics::Marshalling marshal;
    // Use a type that will be translated into LLVM as:
    // { t, t }   struct of 2 eleTy, byval, align 4
    auto structTy =
        mlir::TupleType::get(eleTy.getContext(), mlir::TypeRange{eleTy, eleTy});
    marshal.emplace_back(fir::ReferenceType::get(structTy),
                         AT{/*alignment=*/4, /*byval=*/true});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // i64   pack both floats in a 64-bit GPR
      marshal.emplace_back(mlir::IntegerType::get(eleTy.getContext(), 64),
                           AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // Use a type that will be translated into LLVM as:
      // { t, t }   struct of 2 eleTy, sret, align 4
      auto structTy = mlir::TupleType::get(eleTy.getContext(),
                                           mlir::TypeRange{eleTy, eleTy});
      marshal.emplace_back(fir::ReferenceType::get(structTy),
                           AT{/*alignment=*/4, /*byval=*/false, /*sret=*/true});
    } else {
      typeTodo(sem, loc, "return");
    }
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// i386 (x86 32 bit) Windows target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetI386Win : public GenericTarget<TargetI386Win> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 32;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    // Use a type that will be translated into LLVM as:
    // { t, t }   struct of 2 eleTy, byval, align 4
    auto structTy =
        mlir::TupleType::get(eleTy.getContext(), mlir::TypeRange{eleTy, eleTy});
    marshal.emplace_back(fir::ReferenceType::get(structTy),
                         AT{/*align=*/4, /*byval=*/true});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // i64   pack both floats in a 64-bit GPR
      marshal.emplace_back(mlir::IntegerType::get(eleTy.getContext(), 64),
                           AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // Use a type that will be translated into LLVM as:
      // { double, double }   struct of 2 double, sret, align 8
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/8, /*byval=*/false, /*sret=*/true});
    } else if (sem == &llvm::APFloat::IEEEquad()) {
      // Use a type that will be translated into LLVM as:
      // { fp128, fp128 }   struct of 2 fp128, sret, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/false, /*sret=*/true});
    } else if (sem == &llvm::APFloat::x87DoubleExtended()) {
      // Use a type that will be translated into LLVM as:
      // { x86_fp80, x86_fp80 }   struct of 2 x86_fp80, sret, align 4
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/4, /*byval=*/false, /*sret=*/true});
    } else {
      typeTodo(sem, loc, "return");
    }
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// x86_64 (x86 64 bit) linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetX86_64 : public GenericTarget<TargetX86_64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // <2 x t>   vector of 2 eleTy
      marshal.emplace_back(fir::VectorType::get(2, eleTy), AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // FIXME: In case of SSE register exhaustion, the ABI here may be
      // incorrect since LLVM may pass the real via register and the imaginary
      // part via the stack while the ABI it should be all in register or all
      // in memory. Register occupancy must be analyzed here.
      // two distinct double arguments
      marshal.emplace_back(eleTy, AT{});
      marshal.emplace_back(eleTy, AT{});
    } else if (sem == &llvm::APFloat::x87DoubleExtended()) {
      // Use a type that will be translated into LLVM as:
      // { x86_fp80, x86_fp80 }  struct of 2 fp128, byval, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/true});
    } else if (sem == &llvm::APFloat::IEEEquad()) {
      // Use a type that will be translated into LLVM as:
      // { fp128, fp128 }   struct of 2 fp128, byval, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/true});
    } else {
      typeTodo(sem, loc, "argument");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // <2 x t>   vector of 2 eleTy
      marshal.emplace_back(fir::VectorType::get(2, eleTy), AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // Use a type that will be translated into LLVM as:
      // { double, double }   struct of 2 double
      marshal.emplace_back(mlir::TupleType::get(eleTy.getContext(),
                                                mlir::TypeRange{eleTy, eleTy}),
                           AT{});
    } else if (sem == &llvm::APFloat::x87DoubleExtended()) {
      // { x86_fp80, x86_fp80 }
      marshal.emplace_back(mlir::TupleType::get(eleTy.getContext(),
                                                mlir::TypeRange{eleTy, eleTy}),
                           AT{});
    } else if (sem == &llvm::APFloat::IEEEquad()) {
      // Use a type that will be translated into LLVM as:
      // { fp128, fp128 }   struct of 2 fp128, sret, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/false, /*sret=*/true});
    } else {
      typeTodo(sem, loc, "return");
    }
    return marshal;
  }

  /// X86-64 argument classes from System V ABI version 1.0 section 3.2.3.
  enum ArgClass {
    Integer = 0,
    SSE,
    SSEUp,
    X87,
    X87Up,
    ComplexX87,
    NoClass,
    Memory
  };

  /// Classify an argument type or a field of an aggregate type argument.
  /// See System V ABI version 1.0 section 3.2.3.
  /// The Lo and Hi class are set to the class of the lower eight eightbytes
  /// and upper eight eightbytes on return.
  /// If this is called for an aggregate field, the caller is responsible to
  /// do the post-merge.
  void classify(mlir::Location loc, mlir::Type type, std::uint64_t byteOffset,
                ArgClass &Lo, ArgClass &Hi) const {
    Hi = Lo = ArgClass::NoClass;
    ArgClass &current = byteOffset < 8 ? Lo : Hi;
    // System V AMD64 ABI 3.2.3. version 1.0
    llvm::TypeSwitch<mlir::Type>(type)
        .template Case<mlir::IntegerType>([&](mlir::IntegerType intTy) {
          if (intTy.getWidth() == 128)
            Hi = Lo = ArgClass::Integer;
          else
            current = ArgClass::Integer;
        })
        .template Case<mlir::FloatType>([&](mlir::Type floatTy) {
          const auto *sem = &floatToSemantics(kindMap, floatTy);
          if (sem == &llvm::APFloat::x87DoubleExtended()) {
            Lo = ArgClass::X87;
            Hi = ArgClass::X87Up;
          } else if (sem == &llvm::APFloat::IEEEquad()) {
            Lo = ArgClass::SSE;
            Hi = ArgClass::SSEUp;
          } else {
            current = ArgClass::SSE;
          }
        })
        .template Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
          const auto *sem = &floatToSemantics(kindMap, cmplx.getElementType());
          if (sem == &llvm::APFloat::x87DoubleExtended()) {
            current = ArgClass::ComplexX87;
          } else {
            fir::SequenceType::Shape shape{2};
            classifyArray(loc,
                          fir::SequenceType::get(shape, cmplx.getElementType()),
                          byteOffset, Lo, Hi);
          }
        })
        .template Case<fir::LogicalType>([&](fir::LogicalType logical) {
          if (kindMap.getLogicalBitsize(logical.getFKind()) == 128)
            Hi = Lo = ArgClass::Integer;
          else
            current = ArgClass::Integer;
        })
        .template Case<fir::CharacterType>(
            [&](fir::CharacterType character) { current = ArgClass::Integer; })
        .template Case<fir::SequenceType>([&](fir::SequenceType seqTy) {
          // Array component.
          classifyArray(loc, seqTy, byteOffset, Lo, Hi);
        })
        .template Case<fir::RecordType>([&](fir::RecordType recTy) {
          // Component that is a derived type.
          classifyStruct(loc, recTy, byteOffset, Lo, Hi);
        })
        .template Case<fir::VectorType>([&](fir::VectorType vecTy) {
          // Previously marshalled SSE eight byte for a previous struct
          // argument.
          auto *sem = fir::isa_real(vecTy.getEleTy())
                          ? &floatToSemantics(kindMap, vecTy.getEleTy())
                          : nullptr;
          // Not expecting to hit this todo in standard code (it would
          // require some vector type extension).
          if (!(sem == &llvm::APFloat::IEEEsingle() && vecTy.getLen() <= 2) &&
              !(sem == &llvm::APFloat::IEEEhalf() && vecTy.getLen() <= 4))
            TODO(loc, "passing vector argument to C by value");
          current = SSE;
        })
        .Default([&](mlir::Type ty) {
          if (fir::conformsWithPassByRef(ty))
            current = ArgClass::Integer; // Pointers.
          else
            TODO(loc, "unsupported component type for BIND(C), VALUE derived "
                      "type argument");
        });
  }

  // Classify fields of a derived type starting at \p offset. Returns the new
  // offset. Post-merge is left to the caller.
  std::uint64_t classifyStruct(mlir::Location loc, fir::RecordType recTy,
                               std::uint64_t byteOffset, ArgClass &Lo,
                               ArgClass &Hi) const {
    for (auto component : recTy.getTypeList()) {
      if (byteOffset > 16) {
        // See 3.2.3 p. 1 and note 15. Note that when the offset is bigger
        // than 16 bytes here, it is not a single _m256 and or _m512 entity
        // that could fit in AVX registers.
        Lo = Hi = ArgClass::Memory;
        return byteOffset;
      }
      mlir::Type compType = component.second;
      auto [compSize, compAlign] = fir::getTypeSizeAndAlignmentOrCrash(
          loc, compType, getDataLayout(), kindMap);
      byteOffset = llvm::alignTo(byteOffset, compAlign);
      ArgClass LoComp, HiComp;
      classify(loc, compType, byteOffset, LoComp, HiComp);
      Lo = mergeClass(Lo, LoComp);
      Hi = mergeClass(Hi, HiComp);
      byteOffset = byteOffset + llvm::alignTo(compSize, compAlign);
      if (Lo == ArgClass::Memory || Hi == ArgClass::Memory)
        return byteOffset;
    }
    return byteOffset;
  }

  // Classify fields of a constant size array type starting at \p offset.
  // Returns the new offset. Post-merge is left to the caller.
  void classifyArray(mlir::Location loc, fir::SequenceType seqTy,
                     std::uint64_t byteOffset, ArgClass &Lo,
                     ArgClass &Hi) const {
    mlir::Type eleTy = seqTy.getEleTy();
    const std::uint64_t arraySize = seqTy.getConstantArraySize();
    auto [eleSize, eleAlign] = fir::getTypeSizeAndAlignmentOrCrash(
        loc, eleTy, getDataLayout(), kindMap);
    std::uint64_t eleStorageSize = llvm::alignTo(eleSize, eleAlign);
    for (std::uint64_t i = 0; i < arraySize; ++i) {
      byteOffset = llvm::alignTo(byteOffset, eleAlign);
      if (byteOffset > 16) {
        // See 3.2.3 p. 1 and note 15. Same as in classifyStruct.
        Lo = Hi = ArgClass::Memory;
        return;
      }
      ArgClass LoComp, HiComp;
      classify(loc, eleTy, byteOffset, LoComp, HiComp);
      Lo = mergeClass(Lo, LoComp);
      Hi = mergeClass(Hi, HiComp);
      byteOffset = byteOffset + eleStorageSize;
      if (Lo == ArgClass::Memory || Hi == ArgClass::Memory)
        return;
    }
  }

  // Goes through the previously marshalled arguments and count the
  // register occupancy to check if there are enough registers left.
  bool hasEnoughRegisters(mlir::Location loc, int neededIntRegisters,
                          int neededSSERegisters,
                          const Marshalling &previousArguments) const {
    int availIntRegisters = 6;
    int availSSERegisters = 8;
    for (auto typeAndAttr : previousArguments) {
      const auto &attr = std::get<Attributes>(typeAndAttr);
      if (attr.isByVal())
        continue; // Previous argument passed on the stack.
      ArgClass Lo, Hi;
      Lo = Hi = ArgClass::NoClass;
      classify(loc, std::get<mlir::Type>(typeAndAttr), 0, Lo, Hi);
      // post merge is not needed here since previous aggregate arguments
      // were marshalled into simpler arguments.
      if (Lo == ArgClass::Integer)
        --availIntRegisters;
      else if (Lo == SSE)
        --availSSERegisters;
      if (Hi == ArgClass::Integer)
        --availIntRegisters;
      else if (Hi == ArgClass::SSE)
        --availSSERegisters;
    }
    return availSSERegisters >= neededSSERegisters &&
           availIntRegisters >= neededIntRegisters;
  }

  /// Argument class merging as described in System V ABI 3.2.3 point 4.
  ArgClass mergeClass(ArgClass accum, ArgClass field) const {
    assert((accum != ArgClass::Memory && accum != ArgClass::ComplexX87) &&
           "Invalid accumulated classification during merge.");
    if (accum == field || field == NoClass)
      return accum;
    if (field == ArgClass::Memory)
      return ArgClass::Memory;
    if (accum == NoClass)
      return field;
    if (accum == Integer || field == Integer)
      return ArgClass::Integer;
    if (field == ArgClass::X87 || field == ArgClass::X87Up ||
        field == ArgClass::ComplexX87 || accum == ArgClass::X87 ||
        accum == ArgClass::X87Up)
      return Memory;
    return SSE;
  }

  /// Argument class post merging as described in System V ABI 3.2.3 point 5.
  void postMerge(std::uint64_t byteSize, ArgClass &Lo, ArgClass &Hi) const {
    if (Hi == ArgClass::Memory)
      Lo = ArgClass::Memory;
    if (Hi == ArgClass::X87Up && Lo != ArgClass::X87)
      Lo = ArgClass::Memory;
    if (byteSize > 16 && (Lo != ArgClass::SSE || Hi != ArgClass::SSEUp))
      Lo = ArgClass::Memory;
    if (Hi == ArgClass::SSEUp && Lo != ArgClass::SSE)
      Hi = SSE;
  }

  /// When \p recTy is a one field record type that can be passed
  /// like the field on its own, returns the field type. Returns
  /// a null type otherwise.
  mlir::Type passAsFieldIfOneFieldStruct(fir::RecordType recTy,
                                         bool allowComplex = false) const {
    auto typeList = recTy.getTypeList();
    if (typeList.size() != 1)
      return {};
    mlir::Type fieldType = typeList[0].second;
    if (mlir::isa<mlir::FloatType, mlir::IntegerType, fir::LogicalType>(
            fieldType))
      return fieldType;
    if (allowComplex && mlir::isa<mlir::ComplexType>(fieldType))
      return fieldType;
    if (mlir::isa<fir::CharacterType>(fieldType)) {
      // Only CHARACTER(1) are expected in BIND(C) contexts, which is the only
      // contexts where derived type may be passed in registers.
      assert(mlir::cast<fir::CharacterType>(fieldType).getLen() == 1 &&
             "fir.type value arg character components must have length 1");
      return fieldType;
    }
    // Complex field that needs to be split, or array.
    return {};
  }

  mlir::Type pickLLVMArgType(mlir::Location loc, mlir::MLIRContext *context,
                             ArgClass argClass,
                             std::uint64_t partByteSize) const {
    if (argClass == ArgClass::SSE) {
      if (partByteSize > 16)
        TODO(loc, "passing struct as a real > 128 bits in register");
      // Clang uses vector type when several fp fields are marshalled
      // into a single SSE register (like  <n x smallest fp field> ).
      // It should make no difference from an ABI point of view to just
      // select an fp type of the right size, and it makes things simpler
      // here.
      if (partByteSize > 8)
        return mlir::FloatType::getF128(context);
      if (partByteSize > 4)
        return mlir::FloatType::getF64(context);
      if (partByteSize > 2)
        return mlir::FloatType::getF32(context);
      return mlir::FloatType::getF16(context);
    }
    assert(partByteSize <= 8 &&
           "expect integer part of aggregate argument to fit into eight bytes");
    if (partByteSize > 4)
      return mlir::IntegerType::get(context, 64);
    if (partByteSize > 2)
      return mlir::IntegerType::get(context, 32);
    if (partByteSize > 1)
      return mlir::IntegerType::get(context, 16);
    return mlir::IntegerType::get(context, 8);
  }

  /// Marshal a derived type passed by value like a C struct.
  CodeGenSpecifics::Marshalling
  structArgumentType(mlir::Location loc, fir::RecordType recTy,
                     const Marshalling &previousArguments) const override {
    std::uint64_t byteOffset = 0;
    ArgClass Lo, Hi;
    Lo = Hi = ArgClass::NoClass;
    byteOffset = classifyStruct(loc, recTy, byteOffset, Lo, Hi);
    postMerge(byteOffset, Lo, Hi);
    if (Lo == ArgClass::Memory || Lo == ArgClass::X87 ||
        Lo == ArgClass::ComplexX87)
      return passOnTheStack(loc, recTy, /*isResult=*/false);
    int neededIntRegisters = 0;
    int neededSSERegisters = 0;
    if (Lo == ArgClass::SSE)
      ++neededSSERegisters;
    else if (Lo == ArgClass::Integer)
      ++neededIntRegisters;
    if (Hi == ArgClass::SSE)
      ++neededSSERegisters;
    else if (Hi == ArgClass::Integer)
      ++neededIntRegisters;
    // C struct should not be split into LLVM registers if LLVM codegen is not
    // able to later assign actual registers to all of them (struct passing is
    // all in registers or all on the stack).
    if (!hasEnoughRegisters(loc, neededIntRegisters, neededSSERegisters,
                            previousArguments))
      return passOnTheStack(loc, recTy, /*isResult=*/false);

    if (auto fieldType = passAsFieldIfOneFieldStruct(recTy)) {
      CodeGenSpecifics::Marshalling marshal;
      marshal.emplace_back(fieldType, AT{});
      return marshal;
    }
    if (Hi == ArgClass::NoClass || Hi == ArgClass::SSEUp) {
      // Pass a single integer or floating point argument.
      mlir::Type lowType =
          pickLLVMArgType(loc, recTy.getContext(), Lo, byteOffset);
      CodeGenSpecifics::Marshalling marshal;
      marshal.emplace_back(lowType, AT{});
      return marshal;
    }
    // Split into two integer or floating point arguments.
    // Note that for the first argument, this will always pick i64 or f64 which
    // may be bigger than needed if some struct padding ends the first eight
    // byte (e.g. for `{i32, f64}`). It is valid from an X86-64 ABI and
    // semantic point of view, but it may not match the LLVM IR interface clang
    // would produce for the equivalent C code (the assembly will still be
    // compatible).  This allows keeping the logic simpler here since it
    // avoids computing the "data" size of the Lo part.
    mlir::Type lowType = pickLLVMArgType(loc, recTy.getContext(), Lo, 8u);
    mlir::Type hiType =
        pickLLVMArgType(loc, recTy.getContext(), Hi, byteOffset - 8u);
    CodeGenSpecifics::Marshalling marshal;
    marshal.emplace_back(lowType, AT{});
    marshal.emplace_back(hiType, AT{});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  structReturnType(mlir::Location loc, fir::RecordType recTy) const override {
    std::uint64_t byteOffset = 0;
    ArgClass Lo, Hi;
    Lo = Hi = ArgClass::NoClass;
    byteOffset = classifyStruct(loc, recTy, byteOffset, Lo, Hi);
    mlir::MLIRContext *context = recTy.getContext();
    postMerge(byteOffset, Lo, Hi);
    if (Lo == ArgClass::Memory)
      return passOnTheStack(loc, recTy, /*isResult=*/true);

    // Note that X87/ComplexX87 are passed in memory, but returned via %st0
    // %st1 registers. Here, they are returned as fp80 or {fp80, fp80} by
    // passAsFieldIfOneFieldStruct, and LLVM will use the expected registers.

    // Note that {_Complex long double} is not 100% clear from an ABI
    // perspective because the aggregate post merger rules say it should be
    // passed in memory because it is bigger than 2 eight bytes. This has the
    // funny effect of
    // {_Complex long double} return to be dealt with differently than
    // _Complex long double.

    if (auto fieldType =
            passAsFieldIfOneFieldStruct(recTy, /*allowComplex=*/true)) {
      if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(fieldType))
        return complexReturnType(loc, complexType.getElementType());
      CodeGenSpecifics::Marshalling marshal;
      marshal.emplace_back(fieldType, AT{});
      return marshal;
    }

    if (Hi == ArgClass::NoClass || Hi == ArgClass::SSEUp) {
      // Return a single integer or floating point argument.
      mlir::Type lowType = pickLLVMArgType(loc, context, Lo, byteOffset);
      CodeGenSpecifics::Marshalling marshal;
      marshal.emplace_back(lowType, AT{});
      return marshal;
    }
    // Will be returned in two different registers. Generate {lowTy, HiTy} for
    // the LLVM IR result type.
    CodeGenSpecifics::Marshalling marshal;
    mlir::Type lowType = pickLLVMArgType(loc, context, Lo, 8u);
    mlir::Type hiType = pickLLVMArgType(loc, context, Hi, byteOffset - 8u);
    marshal.emplace_back(mlir::TupleType::get(context, {lowType, hiType}),
                         AT{});
    return marshal;
  }

  /// Marshal an argument that must be passed on the stack.
  CodeGenSpecifics::Marshalling
  passOnTheStack(mlir::Location loc, mlir::Type ty, bool isResult) const {
    CodeGenSpecifics::Marshalling marshal;
    auto sizeAndAlign =
        fir::getTypeSizeAndAlignmentOrCrash(loc, ty, getDataLayout(), kindMap);
    // The stack is always 8 byte aligned (note 14 in 3.2.3).
    unsigned short align =
        std::max(sizeAndAlign.second, static_cast<unsigned short>(8));
    marshal.emplace_back(fir::ReferenceType::get(ty),
                         AT{align, /*byval=*/!isResult, /*sret=*/isResult});
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// x86_64 (x86 64 bit) Windows target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetX86_64Win : public GenericTarget<TargetX86_64Win> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // i64   pack both floats in a 64-bit GPR
      marshal.emplace_back(mlir::IntegerType::get(eleTy.getContext(), 64),
                           AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // Use a type that will be translated into LLVM as:
      // { double, double }   struct of 2 double, byval, align 8
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/8, /*byval=*/true});
    } else if (sem == &llvm::APFloat::IEEEquad() ||
               sem == &llvm::APFloat::x87DoubleExtended()) {
      // Use a type that will be translated into LLVM as:
      // { t, t }   struct of 2 eleTy, byval, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/true});
    } else {
      typeTodo(sem, loc, "argument");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle()) {
      // i64   pack both floats in a 64-bit GPR
      marshal.emplace_back(mlir::IntegerType::get(eleTy.getContext(), 64),
                           AT{});
    } else if (sem == &llvm::APFloat::IEEEdouble()) {
      // Use a type that will be translated into LLVM as:
      // { double, double }   struct of 2 double, sret, align 8
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/8, /*byval=*/false, /*sret=*/true});
    } else if (sem == &llvm::APFloat::IEEEquad() ||
               sem == &llvm::APFloat::x87DoubleExtended()) {
      // Use a type that will be translated into LLVM as:
      // { t, t }   struct of 2 eleTy, sret, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/false, /*sret=*/true});
    } else {
      typeTodo(sem, loc, "return");
    }
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AArch64 linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
// AArch64 procedure call standard:
// https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#parameter-passing
struct TargetAArch64 : public GenericTarget<TargetAArch64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble() ||
        sem == &llvm::APFloat::IEEEquad()) {
      // [2 x t]   array of 2 eleTy
      marshal.emplace_back(fir::SequenceType::get({2}, eleTy), AT{});
    } else {
      typeTodo(sem, loc, "argument");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble() ||
        sem == &llvm::APFloat::IEEEquad()) {
      // Use a type that will be translated into LLVM as:
      // { t, t }   struct of 2 eleTy
      marshal.emplace_back(mlir::TupleType::get(eleTy.getContext(),
                                                mlir::TypeRange{eleTy, eleTy}),
                           AT{});
    } else {
      typeTodo(sem, loc, "return");
    }
    return marshal;
  }

  // Flatten a RecordType::TypeList containing more record types or array type
  static std::optional<std::vector<mlir::Type>>
  flattenTypeList(const RecordType::TypeList &types) {
    std::vector<mlir::Type> flatTypes;
    // The flat list will be at least the same size as the non-flat list.
    flatTypes.reserve(types.size());
    for (auto [c, type] : types) {
      // Flatten record type
      if (auto recTy = mlir::dyn_cast<RecordType>(type)) {
        auto subTypeList = flattenTypeList(recTy.getTypeList());
        if (!subTypeList)
          return std::nullopt;
        llvm::copy(*subTypeList, std::back_inserter(flatTypes));
        continue;
      }

      // Flatten array type
      if (auto seqTy = mlir::dyn_cast<SequenceType>(type)) {
        if (seqTy.hasDynamicExtents())
          return std::nullopt;
        std::size_t n = seqTy.getConstantArraySize();
        auto eleTy = seqTy.getElementType();
        // Flatten array of record types
        if (auto recTy = mlir::dyn_cast<RecordType>(eleTy)) {
          auto subTypeList = flattenTypeList(recTy.getTypeList());
          if (!subTypeList)
            return std::nullopt;
          for (std::size_t i = 0; i < n; ++i)
            llvm::copy(*subTypeList, std::back_inserter(flatTypes));
        } else {
          std::fill_n(std::back_inserter(flatTypes),
                      seqTy.getConstantArraySize(), eleTy);
        }
        continue;
      }

      // Other types are already flat
      flatTypes.push_back(type);
    }
    return flatTypes;
  }

  // Determine if the type is a Homogenous Floating-point Aggregate (HFA). An
  // HFA is a record type with up to 4 floating-point members of the same type.
  static std::optional<int> usedRegsForHFA(fir::RecordType ty) {
    RecordType::TypeList types = ty.getTypeList();
    if (types.empty() || types.size() > 4)
      return std::nullopt;

    std::optional<std::vector<mlir::Type>> flatTypes = flattenTypeList(types);
    if (!flatTypes || flatTypes->size() > 4) {
      return std::nullopt;
    }

    if (!isa_real(flatTypes->front())) {
      return std::nullopt;
    }

    return llvm::all_equal(*flatTypes) ? std::optional<int>{flatTypes->size()}
                                       : std::nullopt;
  }

  struct NRegs {
    int n{0};
    bool isSimd{false};
  };

  NRegs usedRegsForRecordType(mlir::Location loc, fir::RecordType type) const {
    if (std::optional<int> size = usedRegsForHFA(type))
      return {*size, true};

    auto [size, align] = fir::getTypeSizeAndAlignmentOrCrash(
        loc, type, getDataLayout(), kindMap);

    if (size <= 16)
      return {static_cast<int>((size + 7) / 8), false};

    // Pass on the stack, i.e. no registers used
    return {};
  }

  NRegs usedRegsForType(mlir::Location loc, mlir::Type type) const {
    return llvm::TypeSwitch<mlir::Type, NRegs>(type)
        .Case<mlir::IntegerType>([&](auto intTy) {
          return intTy.getWidth() == 128 ? NRegs{2, false} : NRegs{1, false};
        })
        .Case<mlir::FloatType>([&](auto) { return NRegs{1, true}; })
        .Case<mlir::ComplexType>([&](auto) { return NRegs{2, true}; })
        .Case<fir::LogicalType>([&](auto) { return NRegs{1, false}; })
        .Case<fir::CharacterType>([&](auto) { return NRegs{1, false}; })
        .Case<fir::SequenceType>([&](auto ty) {
          assert(ty.getShape().size() == 1 &&
                 "invalid array dimensions in BIND(C)");
          NRegs nregs = usedRegsForType(loc, ty.getEleTy());
          nregs.n *= ty.getShape()[0];
          return nregs;
        })
        .Case<fir::RecordType>(
            [&](auto ty) { return usedRegsForRecordType(loc, ty); })
        .Case<fir::VectorType>([&](auto) {
          TODO(loc, "passing vector argument to C by value is not supported");
          return NRegs{};
        });
  }

  bool hasEnoughRegisters(mlir::Location loc, fir::RecordType type,
                          const Marshalling &previousArguments) const {
    int availIntRegisters = 8;
    int availSIMDRegisters = 8;

    // Check previous arguments to see how many registers are used already
    for (auto [type, attr] : previousArguments) {
      if (availIntRegisters <= 0 || availSIMDRegisters <= 0)
        break;

      if (attr.isByVal())
        continue; // Previous argument passed on the stack

      NRegs nregs = usedRegsForType(loc, type);
      if (nregs.isSimd)
        availSIMDRegisters -= nregs.n;
      else
        availIntRegisters -= nregs.n;
    }

    NRegs nregs = usedRegsForRecordType(loc, type);

    if (nregs.isSimd)
      return nregs.n <= availSIMDRegisters;

    return nregs.n <= availIntRegisters;
  }

  CodeGenSpecifics::Marshalling
  passOnTheStack(mlir::Location loc, mlir::Type ty, bool isResult) const {
    CodeGenSpecifics::Marshalling marshal;
    auto sizeAndAlign =
        fir::getTypeSizeAndAlignmentOrCrash(loc, ty, getDataLayout(), kindMap);
    // The stack is always 8 byte aligned
    unsigned short align =
        std::max(sizeAndAlign.second, static_cast<unsigned short>(8));
    marshal.emplace_back(fir::ReferenceType::get(ty),
                         AT{align, /*byval=*/!isResult, /*sret=*/isResult});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  structType(mlir::Location loc, fir::RecordType type, bool isResult) const {
    NRegs nregs = usedRegsForRecordType(loc, type);

    // If the type needs no registers it must need to be passed on the stack
    if (nregs.n == 0)
      return passOnTheStack(loc, type, isResult);

    CodeGenSpecifics::Marshalling marshal;

    mlir::Type pcsType;
    if (nregs.isSimd) {
      pcsType = type;
    } else {
      pcsType = fir::SequenceType::get(
          nregs.n, mlir::IntegerType::get(type.getContext(), 64));
    }

    marshal.emplace_back(pcsType, AT{});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  structArgumentType(mlir::Location loc, fir::RecordType ty,
                     const Marshalling &previousArguments) const override {
    if (!hasEnoughRegisters(loc, ty, previousArguments)) {
      return passOnTheStack(loc, ty, /*isResult=*/false);
    }

    return structType(loc, ty, /*isResult=*/false);
  }

  CodeGenSpecifics::Marshalling
  structReturnType(mlir::Location loc, fir::RecordType ty) const override {
    return structType(loc, ty, /*isResult=*/true);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PPC64 (AIX 64 bit) target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetPPC64 : public GenericTarget<TargetPPC64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    // two distinct element type arguments (re, im)
    marshal.emplace_back(eleTy, AT{});
    marshal.emplace_back(eleTy, AT{});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    // Use a type that will be translated into LLVM as:
    // { t, t }   struct of 2 element type
    marshal.emplace_back(
        mlir::TupleType::get(eleTy.getContext(), mlir::TypeRange{eleTy, eleTy}),
        AT{});
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// PPC64le linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetPPC64le : public GenericTarget<TargetPPC64le> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    // two distinct element type arguments (re, im)
    marshal.emplace_back(eleTy, AT{});
    marshal.emplace_back(eleTy, AT{});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    // Use a type that will be translated into LLVM as:
    // { t, t }   struct of 2 element type
    marshal.emplace_back(
        mlir::TupleType::get(eleTy.getContext(), mlir::TypeRange{eleTy, eleTy}),
        AT{});
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// sparc (sparc 32 bit) target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetSparc : public GenericTarget<TargetSparc> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 32;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location, mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    CodeGenSpecifics::Marshalling marshal;
    // Use a type that will be translated into LLVM as:
    // { t, t }   struct of 2 eleTy
    auto structTy =
        mlir::TupleType::get(eleTy.getContext(), mlir::TypeRange{eleTy, eleTy});
    marshal.emplace_back(fir::ReferenceType::get(structTy), AT{});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    assert(fir::isa_real(eleTy));
    CodeGenSpecifics::Marshalling marshal;
    // Use a type that will be translated into LLVM as:
    // { t, t }   struct of 2 eleTy, byval
    auto structTy =
        mlir::TupleType::get(eleTy.getContext(), mlir::TypeRange{eleTy, eleTy});
    marshal.emplace_back(fir::ReferenceType::get(structTy),
                         AT{/*alignment=*/0, /*byval=*/true});
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// sparcv9 (sparc 64 bit) target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetSparcV9 : public GenericTarget<TargetSparcV9> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble()) {
      // two distinct float, double arguments
      marshal.emplace_back(eleTy, AT{});
      marshal.emplace_back(eleTy, AT{});
    } else if (sem == &llvm::APFloat::IEEEquad()) {
      // Use a type that will be translated into LLVM as:
      // { fp128, fp128 }   struct of 2 fp128, byval, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/true});
    } else {
      typeTodo(sem, loc, "argument");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    // Use a type that will be translated into LLVM as:
    // { eleTy, eleTy }   struct of 2 eleTy
    marshal.emplace_back(
        mlir::TupleType::get(eleTy.getContext(), mlir::TypeRange{eleTy, eleTy}),
        AT{});
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// RISCV64 linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetRISCV64 : public GenericTarget<TargetRISCV64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble()) {
      // Two distinct element type arguments (re, im)
      marshal.emplace_back(eleTy, AT{});
      marshal.emplace_back(eleTy, AT{});
    } else {
      typeTodo(sem, loc, "argument");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble()) {
      // Use a type that will be translated into LLVM as:
      // { t, t }   struct of 2 eleTy, byVal
      marshal.emplace_back(mlir::TupleType::get(eleTy.getContext(),
                                                mlir::TypeRange{eleTy, eleTy}),
                           AT{/*alignment=*/0, /*byval=*/true});
    } else {
      typeTodo(sem, loc, "return");
    }
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// AMDGPU linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetAMDGPU : public GenericTarget<TargetAMDGPU> {
  using GenericTarget::GenericTarget;

  // Default size (in bits) of the index type for strings.
  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    TODO(loc, "handle complex argument types");
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    TODO(loc, "handle complex return types");
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// NVPTX linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetNVPTX : public GenericTarget<TargetNVPTX> {
  using GenericTarget::GenericTarget;

  // Default size (in bits) of the index type for strings.
  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    TODO(loc, "handle complex argument types");
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    TODO(loc, "handle complex return types");
    return marshal;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// LoongArch64 linux target specifics.
//===----------------------------------------------------------------------===//

namespace {
struct TargetLoongArch64 : public GenericTarget<TargetLoongArch64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;
  static constexpr int GRLen = defaultWidth; /* eight bytes */
  static constexpr int GRLenInChar = GRLen / 8;
  static constexpr int FRLen = defaultWidth; /* eight bytes */

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble()) {
      // Two distinct element type arguments (re, im)
      marshal.emplace_back(eleTy, AT{});
      marshal.emplace_back(eleTy, AT{});
    } else if (sem == &llvm::APFloat::IEEEquad()) {
      // Use a type that will be translated into LLVM as:
      // { fp128, fp128 }   struct of 2 fp128, byval
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/true});
    } else {
      typeTodo(sem, loc, "argument");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  complexReturnType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble()) {
      // Use a type that will be translated into LLVM as:
      // { t, t }   struct of 2 eleTy, byVal
      marshal.emplace_back(mlir::TupleType::get(eleTy.getContext(),
                                                mlir::TypeRange{eleTy, eleTy}),
                           AT{/*alignment=*/0, /*byval=*/true});
    } else if (sem == &llvm::APFloat::IEEEquad()) {
      // Use a type that will be translated into LLVM as:
      // { fp128, fp128 }   struct of 2 fp128, sret, align 16
      marshal.emplace_back(
          fir::ReferenceType::get(mlir::TupleType::get(
              eleTy.getContext(), mlir::TypeRange{eleTy, eleTy})),
          AT{/*align=*/16, /*byval=*/false, /*sret=*/true});
    } else {
      typeTodo(sem, loc, "return");
    }
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  integerArgumentType(mlir::Location loc,
                      mlir::IntegerType argTy) const override {
    if (argTy.getWidth() == 32) {
      // LA64 LP64D ABI requires unsigned 32 bit integers to be sign extended.
      // Therefore, Flang also follows it if a function needs to be
      // interoperable with C.
      //
      // Currently, it only adds `signext` attribute to the dummy arguments and
      // return values in the function signatures, but it does not add the
      // corresponding attribute to the actual arguments and return values in
      // `fir.call` instruction. Thanks to LLVM's integration of all these
      // attributes, the modification is still effective.
      CodeGenSpecifics::Marshalling marshal;
      AT::IntegerExtension intExt = AT::IntegerExtension::Sign;
      marshal.emplace_back(argTy, AT{/*alignment=*/0, /*byval=*/false,
                                     /*sret=*/false, /*append=*/false,
                                     /*intExt=*/intExt});
      return marshal;
    }

    return GenericTarget::integerArgumentType(loc, argTy);
  }

  /// Flatten non-basic types, resulting in an array of types containing only
  /// `IntegerType` and `FloatType`.
  llvm::SmallVector<mlir::Type> flattenTypeList(mlir::Location loc,
                                                const mlir::Type type) const {
    llvm::SmallVector<mlir::Type> flatTypes;

    llvm::TypeSwitch<mlir::Type>(type)
        .template Case<mlir::IntegerType>([&](mlir::IntegerType intTy) {
          if (intTy.getWidth() != 0)
            flatTypes.push_back(intTy);
        })
        .template Case<mlir::FloatType>([&](mlir::FloatType floatTy) {
          if (floatTy.getWidth() != 0)
            flatTypes.push_back(floatTy);
        })
        .template Case<mlir::ComplexType>([&](mlir::ComplexType cmplx) {
          const auto *sem = &floatToSemantics(kindMap, cmplx.getElementType());
          if (sem == &llvm::APFloat::IEEEsingle() ||
              sem == &llvm::APFloat::IEEEdouble() ||
              sem == &llvm::APFloat::IEEEquad())
            std::fill_n(std::back_inserter(flatTypes), 2,
                        cmplx.getElementType());
          else
            TODO(loc, "unsupported complex type(not IEEEsingle, IEEEdouble, "
                      "IEEEquad) as a structure component for BIND(C), "
                      "VALUE derived type argument and type return");
        })
        .template Case<fir::LogicalType>([&](fir::LogicalType logicalTy) {
          const unsigned width =
              kindMap.getLogicalBitsize(logicalTy.getFKind());
          if (width != 0)
            flatTypes.push_back(
                mlir::IntegerType::get(type.getContext(), width));
        })
        .template Case<fir::CharacterType>([&](fir::CharacterType charTy) {
          assert(kindMap.getCharacterBitsize(charTy.getFKind()) <= 8 &&
                 "the bit size of characterType as an interoperable type must "
                 "not exceed 8");
          for (unsigned i = 0; i < charTy.getLen(); ++i)
            flatTypes.push_back(mlir::IntegerType::get(type.getContext(), 8));
        })
        .template Case<fir::SequenceType>([&](fir::SequenceType seqTy) {
          if (!seqTy.hasDynamicExtents()) {
            const std::uint64_t numOfEle = seqTy.getConstantArraySize();
            mlir::Type eleTy = seqTy.getEleTy();
            if (!mlir::isa<mlir::IntegerType, mlir::FloatType>(eleTy)) {
              llvm::SmallVector<mlir::Type> subTypeList =
                  flattenTypeList(loc, eleTy);
              if (subTypeList.size() != 0)
                for (std::uint64_t i = 0; i < numOfEle; ++i)
                  llvm::copy(subTypeList, std::back_inserter(flatTypes));
            } else {
              std::fill_n(std::back_inserter(flatTypes), numOfEle, eleTy);
            }
          } else
            TODO(loc, "unsupported dynamic extent sequence type as a structure "
                      "component for BIND(C), "
                      "VALUE derived type argument and type return");
        })
        .template Case<fir::RecordType>([&](fir::RecordType recTy) {
          for (auto &component : recTy.getTypeList()) {
            mlir::Type eleTy = component.second;
            llvm::SmallVector<mlir::Type> subTypeList =
                flattenTypeList(loc, eleTy);
            if (subTypeList.size() != 0)
              llvm::copy(subTypeList, std::back_inserter(flatTypes));
          }
        })
        .template Case<fir::VectorType>([&](fir::VectorType vecTy) {
          auto sizeAndAlign = fir::getTypeSizeAndAlignmentOrCrash(
              loc, vecTy, getDataLayout(), kindMap);
          if (sizeAndAlign.first == 2 * GRLenInChar)
            flatTypes.push_back(
                mlir::IntegerType::get(type.getContext(), 2 * GRLen));
          else
            TODO(loc, "unsupported vector width(must be 128 bits)");
        })
        .Default([&](mlir::Type ty) {
          if (fir::conformsWithPassByRef(ty))
            flatTypes.push_back(
                mlir::IntegerType::get(type.getContext(), GRLen));
          else
            TODO(loc, "unsupported component type for BIND(C), VALUE derived "
                      "type argument and type return");
        });

    return flatTypes;
  }

  /// Determine if a struct is eligible to be passed in FARs (and GARs) (i.e.,
  /// when flattened it contains a single fp value, fp+fp, or int+fp of
  /// appropriate size).
  bool detectFARsEligibleStruct(mlir::Location loc, fir::RecordType recTy,
                                mlir::Type &field1Ty,
                                mlir::Type &field2Ty) const {
    field1Ty = field2Ty = nullptr;
    llvm::SmallVector<mlir::Type> flatTypes = flattenTypeList(loc, recTy);
    size_t flatSize = flatTypes.size();

    // Cannot be eligible if the number of flattened types is equal to 0 or
    // greater than 2.
    if (flatSize == 0 || flatSize > 2)
      return false;

    bool isFirstAvaliableFloat = false;

    assert((mlir::isa<mlir::IntegerType, mlir::FloatType>(flatTypes[0])) &&
           "Type must be integerType or floatType after flattening");
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(flatTypes[0])) {
      const unsigned Size = floatTy.getWidth();
      // Can't be eligible if larger than the FP registers. Half precision isn't
      // currently supported on LoongArch and the ABI hasn't been confirmed, so
      // default to the integer ABI in that case.
      if (Size > FRLen || Size < 32)
        return false;
      isFirstAvaliableFloat = true;
      field1Ty = floatTy;
    } else if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(flatTypes[0])) {
      if (intTy.getWidth() > GRLen)
        return false;
      field1Ty = intTy;
    }

    // flatTypes has two elements
    if (flatSize == 2) {
      assert((mlir::isa<mlir::IntegerType, mlir::FloatType>(flatTypes[1])) &&
             "Type must be integerType or floatType after flattening");
      if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(flatTypes[1])) {
        const unsigned Size = floatTy.getWidth();
        if (Size > FRLen || Size < 32)
          return false;
        field2Ty = floatTy;
        return true;
      } else if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(flatTypes[1])) {
        // Can't be eligible if an integer type was already found (int+int pairs
        // are not eligible).
        if (!isFirstAvaliableFloat)
          return false;
        if (intTy.getWidth() > GRLen)
          return false;
        field2Ty = intTy;
        return true;
      }
    }

    // return isFirstAvaliableFloat if flatTypes only has one element
    return isFirstAvaliableFloat;
  }

  bool checkTypeHasEnoughRegs(mlir::Location loc, int &GARsLeft, int &FARsLeft,
                              const mlir::Type type) const {
    if (!type)
      return true;

    llvm::TypeSwitch<mlir::Type>(type)
        .template Case<mlir::IntegerType>([&](mlir::IntegerType intTy) {
          const unsigned width = intTy.getWidth();
          if (width > 128)
            TODO(loc,
                 "integerType with width exceeding 128 bits is unsupported");
          if (width == 0)
            return;
          if (width <= GRLen)
            --GARsLeft;
          else if (width <= 2 * GRLen)
            GARsLeft = GARsLeft - 2;
        })
        .template Case<mlir::FloatType>([&](mlir::FloatType floatTy) {
          const unsigned width = floatTy.getWidth();
          if (width > 128)
            TODO(loc, "floatType with width exceeding 128 bits is unsupported");
          if (width == 0)
            return;
          if (width == 32 || width == 64)
            --FARsLeft;
          else if (width <= GRLen)
            --GARsLeft;
          else if (width <= 2 * GRLen)
            GARsLeft = GARsLeft - 2;
        })
        .Default([&](mlir::Type ty) {
          if (fir::conformsWithPassByRef(ty))
            --GARsLeft; // Pointers.
          else
            TODO(loc, "unsupported component type for BIND(C), VALUE derived "
                      "type argument and type return");
        });

    return GARsLeft >= 0 && FARsLeft >= 0;
  }

  bool hasEnoughRegisters(mlir::Location loc, int GARsLeft, int FARsLeft,
                          const Marshalling &previousArguments,
                          const mlir::Type &field1Ty,
                          const mlir::Type &field2Ty) const {
    for (auto &typeAndAttr : previousArguments) {
      const auto &attr = std::get<Attributes>(typeAndAttr);
      if (attr.isByVal()) {
        // Previous argument passed on the stack, and its address is passed in
        // GAR.
        --GARsLeft;
        continue;
      }

      // Previous aggregate arguments were marshalled into simpler arguments.
      const auto &type = std::get<mlir::Type>(typeAndAttr);
      llvm::SmallVector<mlir::Type> flatTypes = flattenTypeList(loc, type);

      for (auto &flatTy : flatTypes) {
        if (!checkTypeHasEnoughRegs(loc, GARsLeft, FARsLeft, flatTy))
          return false;
      }
    }

    if (!checkTypeHasEnoughRegs(loc, GARsLeft, FARsLeft, field1Ty))
      return false;
    if (!checkTypeHasEnoughRegs(loc, GARsLeft, FARsLeft, field2Ty))
      return false;
    return true;
  }

  /// LoongArch64 subroutine calling sequence ABI in:
  /// https://github.com/loongson/la-abi-specs/blob/release/lapcs.adoc#subroutine-calling-sequence
  CodeGenSpecifics::Marshalling
  classifyStruct(mlir::Location loc, fir::RecordType recTy, int GARsLeft,
                 int FARsLeft, bool isResult,
                 const Marshalling &previousArguments) const {
    CodeGenSpecifics::Marshalling marshal;

    auto [recSize, recAlign] = fir::getTypeSizeAndAlignmentOrCrash(
        loc, recTy, getDataLayout(), kindMap);
    mlir::MLIRContext *context = recTy.getContext();

    if (recSize == 0) {
      TODO(loc, "unsupported empty struct type for BIND(C), "
                "VALUE derived type argument and type return");
    }

    if (recSize > 2 * GRLenInChar) {
      marshal.emplace_back(
          fir::ReferenceType::get(recTy),
          AT{recAlign, /*byval=*/!isResult, /*sret=*/isResult});
      return marshal;
    }

    // Pass by FARs(and GARs)
    mlir::Type field1Ty = nullptr, field2Ty = nullptr;
    if (detectFARsEligibleStruct(loc, recTy, field1Ty, field2Ty) &&
        hasEnoughRegisters(loc, GARsLeft, FARsLeft, previousArguments, field1Ty,
                           field2Ty)) {
      if (!isResult) {
        if (field1Ty)
          marshal.emplace_back(field1Ty, AT{});
        if (field2Ty)
          marshal.emplace_back(field2Ty, AT{});
      } else {
        // field1Ty is always preferred over field2Ty for assignment, so there
        // will never be a case where field1Ty == nullptr and field2Ty !=
        // nullptr.
        if (field1Ty && !field2Ty)
          marshal.emplace_back(field1Ty, AT{});
        else if (field1Ty && field2Ty)
          marshal.emplace_back(
              mlir::TupleType::get(context,
                                   mlir::TypeRange{field1Ty, field2Ty}),
              AT{/*alignment=*/0, /*byval=*/true});
      }
      return marshal;
    }

    if (recSize <= GRLenInChar) {
      marshal.emplace_back(mlir::IntegerType::get(context, GRLen), AT{});
      return marshal;
    }

    if (recAlign == 2 * GRLenInChar) {
      marshal.emplace_back(mlir::IntegerType::get(context, 2 * GRLen), AT{});
      return marshal;
    }

    // recSize > GRLenInChar && recSize <= 2 * GRLenInChar
    marshal.emplace_back(
        fir::SequenceType::get({2}, mlir::IntegerType::get(context, GRLen)),
        AT{});
    return marshal;
  }

  /// Marshal a derived type passed by value like a C struct.
  CodeGenSpecifics::Marshalling
  structArgumentType(mlir::Location loc, fir::RecordType recTy,
                     const Marshalling &previousArguments) const override {
    int GARsLeft = 8;
    int FARsLeft = FRLen ? 8 : 0;

    return classifyStruct(loc, recTy, GARsLeft, FARsLeft, /*isResult=*/false,
                          previousArguments);
  }

  CodeGenSpecifics::Marshalling
  structReturnType(mlir::Location loc, fir::RecordType recTy) const override {
    // The rules for return and argument types are the same.
    int GARsLeft = 2;
    int FARsLeft = FRLen ? 2 : 0;
    return classifyStruct(loc, recTy, GARsLeft, FARsLeft, /*isResult=*/true,
                          {});
  }
};
} // namespace

// Instantiate the overloaded target instance based on the triple value.
// TODO: Add other targets to this file as needed.
std::unique_ptr<fir::CodeGenSpecifics>
fir::CodeGenSpecifics::get(mlir::MLIRContext *ctx, llvm::Triple &&trp,
                           KindMapping &&kindMap, llvm::StringRef targetCPU,
                           mlir::LLVM::TargetFeaturesAttr targetFeatures,
                           const mlir::DataLayout &dl) {
  switch (trp.getArch()) {
  default:
    break;
  case llvm::Triple::ArchType::x86:
    if (trp.isOSWindows())
      return std::make_unique<TargetI386Win>(ctx, std::move(trp),
                                             std::move(kindMap), targetCPU,
                                             targetFeatures, dl);
    else
      return std::make_unique<TargetI386>(ctx, std::move(trp),
                                          std::move(kindMap), targetCPU,
                                          targetFeatures, dl);
  case llvm::Triple::ArchType::x86_64:
    if (trp.isOSWindows())
      return std::make_unique<TargetX86_64Win>(ctx, std::move(trp),
                                               std::move(kindMap), targetCPU,
                                               targetFeatures, dl);
    else
      return std::make_unique<TargetX86_64>(ctx, std::move(trp),
                                            std::move(kindMap), targetCPU,
                                            targetFeatures, dl);
  case llvm::Triple::ArchType::aarch64:
    return std::make_unique<TargetAArch64>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::ppc64:
    return std::make_unique<TargetPPC64>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::ppc64le:
    return std::make_unique<TargetPPC64le>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::sparc:
    return std::make_unique<TargetSparc>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::sparcv9:
    return std::make_unique<TargetSparcV9>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::riscv64:
    return std::make_unique<TargetRISCV64>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::amdgcn:
    return std::make_unique<TargetAMDGPU>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::nvptx64:
    return std::make_unique<TargetNVPTX>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  case llvm::Triple::ArchType::loongarch64:
    return std::make_unique<TargetLoongArch64>(
        ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);
  }
  TODO(mlir::UnknownLoc::get(ctx), "target not implemented");
}

std::unique_ptr<fir::CodeGenSpecifics> fir::CodeGenSpecifics::get(
    mlir::MLIRContext *ctx, llvm::Triple &&trp, KindMapping &&kindMap,
    llvm::StringRef targetCPU, mlir::LLVM::TargetFeaturesAttr targetFeatures,
    const mlir::DataLayout &dl, llvm::StringRef tuneCPU) {
  std::unique_ptr<fir::CodeGenSpecifics> CGS = fir::CodeGenSpecifics::get(
      ctx, std::move(trp), std::move(kindMap), targetCPU, targetFeatures, dl);

  CGS->tuneCPU = tuneCPU;
  return CGS;
}
