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
  if (auto ty = type.dyn_cast<fir::RealType>())
    return kindMap.getFloatSemantics(ty.getFKind());
  return type.cast<mlir::FloatType>().getFloatSemantics();
}

static void typeTodo(const llvm::fltSemantics *sem, mlir::Location loc,
                     std::string context) {
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

/// Return the size and alignment of FIR types.
/// TODO: consider moving this to a DataLayoutTypeInterface implementation
/// for FIR types. It should first be ensured that it is OK to open the gate of
/// target dependent type size inquiries in lowering. It would also not be
/// straightforward given the need for a kind map that would need to be
/// converted in terms of mlir::DataLayoutEntryKey.
static std::pair<std::uint64_t, unsigned short>
getSizeAndAlignment(mlir::Location loc, mlir::Type ty,
                    const mlir::DataLayout &dl,
                    const fir::KindMapping &kindMap) {
  if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::ComplexType>(ty)) {
    llvm::TypeSize size = dl.getTypeSize(ty);
    unsigned short alignment = dl.getTypeABIAlignment(ty);
    return {size, alignment};
  }
  if (auto firCmplx = mlir::dyn_cast<fir::ComplexType>(ty)) {
    auto [floatSize, floatAlign] =
        getSizeAndAlignment(loc, firCmplx.getEleType(kindMap), dl, kindMap);
    return {llvm::alignTo(floatSize, floatAlign) + floatSize, floatAlign};
  }
  if (auto real = mlir::dyn_cast<fir::RealType>(ty))
    return getSizeAndAlignment(loc, real.getFloatType(kindMap), dl, kindMap);

  if (auto seqTy = mlir::dyn_cast<fir::SequenceType>(ty)) {
    auto [eleSize, eleAlign] =
        getSizeAndAlignment(loc, seqTy.getEleTy(), dl, kindMap);

    std::uint64_t size =
        llvm::alignTo(eleSize, eleAlign) * seqTy.getConstantArraySize();
    return {size, eleAlign};
  }
  if (auto recTy = mlir::dyn_cast<fir::RecordType>(ty)) {
    std::uint64_t size = 0;
    unsigned short align = 1;
    for (auto component : recTy.getTypeList()) {
      auto [compSize, compAlign] =
          getSizeAndAlignment(loc, component.second, dl, kindMap);
      size =
          llvm::alignTo(size, compAlign) + llvm::alignTo(compSize, compAlign);
      align = std::max(align, compAlign);
    }
    return {size, align};
  }
  if (auto logical = mlir::dyn_cast<fir::LogicalType>(ty)) {
    mlir::Type intTy = mlir::IntegerType::get(
        logical.getContext(), kindMap.getLogicalBitsize(logical.getFKind()));
    return getSizeAndAlignment(loc, intTy, dl, kindMap);
  }
  if (auto character = mlir::dyn_cast<fir::CharacterType>(ty)) {
    mlir::Type intTy = mlir::IntegerType::get(
        character.getContext(),
        kindMap.getCharacterBitsize(character.getFKind()));
    return getSizeAndAlignment(loc, intTy, dl, kindMap);
  }
  TODO(loc, "computing size of a component");
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

  Marshalling boxcharArgumentType(mlir::Type eleTy, bool sret) const override {
    CodeGenSpecifics::Marshalling marshal;
    auto idxTy = mlir::IntegerType::get(eleTy.getContext(), S::defaultWidth);
    auto ptrTy = fir::ReferenceType::get(eleTy);
    marshal.emplace_back(ptrTy, AT{});
    // Return value arguments are grouped as a pair. Others are passed in a
    // split format with all pointers first (in the declared position) and all
    // LEN arguments appended after all of the dummy arguments.
    // NB: Other conventions/ABIs can/should be supported via options.
    marshal.emplace_back(idxTy, AT{/*alignment=*/0, /*byval=*/false,
                                   /*sret=*/sret, /*append=*/!sret});
    return marshal;
  }

  CodeGenSpecifics::Marshalling
  structArgumentType(mlir::Location loc, fir::RecordType,
                     const Marshalling &) const override {
    TODO(loc, "passing VALUE BIND(C) derived type for this target");
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
        .template Case<mlir::FloatType, fir::RealType>([&](mlir::Type floatTy) {
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
        .template Case<fir::ComplexType>([&](fir::ComplexType cmplx) {
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
      auto [compSize, compAlign] =
          getSizeAndAlignment(loc, compType, getDataLayout(), kindMap);
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
    auto [eleSize, eleAlign] =
        getSizeAndAlignment(loc, eleTy, getDataLayout(), kindMap);
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
      return passOnTheStack(loc, recTy);
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
      return passOnTheStack(loc, recTy);
    // TODO, marshal the struct into registers.
    TODO(loc, "passing BIND(C), VALUE derived type in registers on X86-64");
  }

  /// Marshal an argument that must be passed on the stack.
  CodeGenSpecifics::Marshalling passOnTheStack(mlir::Location loc,
                                               mlir::Type ty) const {
    CodeGenSpecifics::Marshalling marshal;
    auto sizeAndAlign = getSizeAndAlignment(loc, ty, getDataLayout(), kindMap);
    // The stack is always 8 byte aligned (note 14 in 3.2.3).
    unsigned short align =
        std::max(sizeAndAlign.second, static_cast<unsigned short>(8));
    marshal.emplace_back(fir::ReferenceType::get(ty),
                         AT{align, /*byval=*/true, /*sret=*/false});
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
struct TargetAArch64 : public GenericTarget<TargetAArch64> {
  using GenericTarget::GenericTarget;

  static constexpr int defaultWidth = 64;

  CodeGenSpecifics::Marshalling
  complexArgumentType(mlir::Location loc, mlir::Type eleTy) const override {
    CodeGenSpecifics::Marshalling marshal;
    const auto *sem = &floatToSemantics(kindMap, eleTy);
    if (sem == &llvm::APFloat::IEEEsingle() ||
        sem == &llvm::APFloat::IEEEdouble()) {
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
        sem == &llvm::APFloat::IEEEdouble()) {
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

// Instantiate the overloaded target instance based on the triple value.
// TODO: Add other targets to this file as needed.
std::unique_ptr<fir::CodeGenSpecifics>
fir::CodeGenSpecifics::get(mlir::MLIRContext *ctx, llvm::Triple &&trp,
                           KindMapping &&kindMap, const mlir::DataLayout &dl) {
  switch (trp.getArch()) {
  default:
    break;
  case llvm::Triple::ArchType::x86:
    if (trp.isOSWindows())
      return std::make_unique<TargetI386Win>(ctx, std::move(trp),
                                             std::move(kindMap), dl);
    else
      return std::make_unique<TargetI386>(ctx, std::move(trp),
                                          std::move(kindMap), dl);
  case llvm::Triple::ArchType::x86_64:
    if (trp.isOSWindows())
      return std::make_unique<TargetX86_64Win>(ctx, std::move(trp),
                                               std::move(kindMap), dl);
    else
      return std::make_unique<TargetX86_64>(ctx, std::move(trp),
                                            std::move(kindMap), dl);
  case llvm::Triple::ArchType::aarch64:
    return std::make_unique<TargetAArch64>(ctx, std::move(trp),
                                           std::move(kindMap), dl);
  case llvm::Triple::ArchType::ppc64:
    return std::make_unique<TargetPPC64>(ctx, std::move(trp),
                                         std::move(kindMap), dl);
  case llvm::Triple::ArchType::ppc64le:
    return std::make_unique<TargetPPC64le>(ctx, std::move(trp),
                                           std::move(kindMap), dl);
  case llvm::Triple::ArchType::sparc:
    return std::make_unique<TargetSparc>(ctx, std::move(trp),
                                         std::move(kindMap), dl);
  case llvm::Triple::ArchType::sparcv9:
    return std::make_unique<TargetSparcV9>(ctx, std::move(trp),
                                           std::move(kindMap), dl);
  case llvm::Triple::ArchType::riscv64:
    return std::make_unique<TargetRISCV64>(ctx, std::move(trp),
                                           std::move(kindMap), dl);
  case llvm::Triple::ArchType::amdgcn:
    return std::make_unique<TargetAMDGPU>(ctx, std::move(trp),
                                          std::move(kindMap), dl);
  case llvm::Triple::ArchType::nvptx64:
    return std::make_unique<TargetNVPTX>(ctx, std::move(trp),
                                         std::move(kindMap), dl);
  case llvm::Triple::ArchType::loongarch64:
    return std::make_unique<TargetLoongArch64>(ctx, std::move(trp),
                                               std::move(kindMap), dl);
  }
  TODO(mlir::UnknownLoc::get(ctx), "target not implemented");
}
