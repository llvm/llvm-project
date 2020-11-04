//===-- TypeConverter.h -- type conversion ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPTIMIZER_CODEGEN_TYPECONVERTER_H
#define OPTIMIZER_CODEGEN_TYPECONVERTER_H

namespace fir {

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::ModuleOp module)
      : mlir::LLVMTypeConverter(module.getContext()),
        kindMapping(*getKindMapping(module)), uniquer(*getNameUniquer(module)),
        specifics(CodeGenSpecifics::get(module.getContext(),
                                        *getTargetTriple(module),
                                        *getKindMapping(module))) {
    LLVM_DEBUG(llvm::dbgs() << "FIR type converter\n");

    // Each conversion should return a value of type mlir::LLVM::LLVMType.
    addConversion([&](BoxType box) { return convertBoxType(box); });
    addConversion([&](BoxCharType boxchar) {
      LLVM_DEBUG(llvm::dbgs() << "type convert: " << boxchar << '\n');
      return unwrap(
          convertType(specifics->boxcharMemoryType(boxchar.getEleTy())));
    });
    addConversion(
        [&](BoxProcType boxproc) { return convertBoxProcType(boxproc); });
    addConversion(
        [&](fir::CharacterType charTy) { return convertCharType(charTy); });
    addConversion(
        [&](mlir::ComplexType cmplx) { return convertComplexType(cmplx); });
    addConversion(
        [&](fir::ComplexType cmplx) { return convertComplexType(cmplx); });
    addConversion(
        [&](fir::RecordType derived) { return convertRecordType(derived); });
    addConversion([&](fir::FieldType field) {
      return mlir::LLVM::LLVMType::getInt32Ty(field.getContext());
    });
    addConversion([&](HeapType heap) { return convertPointerLike(heap); });
    addConversion([&](fir::IntegerType intTy) {
      return mlir::LLVM::LLVMType::getIntNTy(
          &getContext(), kindMapping.getIntegerBitsize(intTy.getFKind()));
    });
    addConversion([&](LenType field) {
      return mlir::LLVM::LLVMType::getInt32Ty(field.getContext());
    });
    addConversion([&](fir::LogicalType boolTy) {
      return mlir::LLVM::LLVMType::getIntNTy(
          &getContext(), kindMapping.getLogicalBitsize(boolTy.getFKind()));
    });
    addConversion(
        [&](fir::PointerType pointer) { return convertPointerLike(pointer); });
    addConversion(
        [&](fir::RealType real) { return convertRealType(real.getFKind()); });
    addConversion(
        [&](fir::ReferenceType ref) { return convertPointerLike(ref); });
    addConversion(
        [&](SequenceType sequence) { return convertSequenceType(sequence); });
    addConversion([&](TypeDescType tdesc) {
      return convertTypeDescType(tdesc.getContext());
    });
    addConversion([&](fir::VectorType vecTy) {
      return mlir::LLVM::LLVMType::getVectorTy(
          unwrap(convertType(vecTy.getEleTy())), vecTy.getLen());
    });
    addConversion([&](mlir::TupleType tuple) {
      LLVM_DEBUG(llvm::dbgs() << "type convert: " << tuple << '\n');
      SmallVector<mlir::Type, 8> inMembers;
      tuple.getFlattenedTypes(inMembers);
      SmallVector<mlir::LLVM::LLVMType, 8> members;
      for (auto mem : inMembers)
        members.push_back(convertType(mem).cast<mlir::LLVM::LLVMType>());
      return mlir::LLVM::LLVMType::getStructTy(&getContext(), members);
    });
    addConversion([&](mlir::NoneType none) {
      return mlir::LLVM::LLVMStructType::getLiteral(none.getContext(),
                                                    llvm::None);
    });

    // FIXME: https://reviews.llvm.org/D82831 introduced an automatic
    // materialization of conversion around function calls that is not working
    // well with fir lowering to llvm (incorrect llvm.mlir.cast are inserted).
    // Workaround until better analysis: register a handler that does not insert
    // any conversions.
    addSourceMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> llvm::Optional<mlir::Value> {
          if (inputs.size() != 1)
            return llvm::None;
          return inputs[0];
        });
    // Similar FIXME workaround here (needed for compare.fir/select-type.fir
    // tests).
    addTargetMaterialization(
        [&](mlir::OpBuilder &builder, mlir::Type resultType,
            mlir::ValueRange inputs,
            mlir::Location loc) -> llvm::Optional<mlir::Value> {
          if (inputs.size() != 1)
            return llvm::None;
          return inputs[0];
        });
  }

  // i32 is used here because LLVM wants i32 constants when indexing into struct
  // types. Indexing into other aggregate types is more flexible.
  mlir::LLVM::LLVMType offsetType() {
    return mlir::LLVM::LLVMType::getInt32Ty(&getContext());
  }

  // i64 can be used to index into aggregates like arrays
  mlir::LLVM::LLVMType indexType() {
    return mlir::LLVM::LLVMType::getInt64Ty(&getContext());
  }

  // TODO
  bool requiresExtendedDesc() { return false; }

  // Magic value to indicate we do not know the rank of an entity, either
  // because it is assumed rank or because we have not determined it yet.
  static constexpr int unknownRank() { return -1; }
  // This corresponds to the descriptor as defined ISO_Fortran_binding.h and the
  // addendum defined in descriptor.h.
  mlir::LLVM::LLVMType convertBoxType(BoxType box, int rank = unknownRank()) {
    // (buffer*, ele-size, rank, type-descriptor, attribute, [dims])
    SmallVector<mlir::LLVM::LLVMType, 6> parts;
    mlir::Type ele = box.getEleTy();
    // remove fir.heap/fir.ref/fir.ptr
    if (auto removeIndirection = fir::dyn_cast_ptrEleTy(ele))
      ele = removeIndirection;
    auto eleTy = unwrap(convertType(ele));
    // buffer*
    if (ele.isa<SequenceType>() && eleTy.isPointerTy())
      parts.push_back(eleTy);
    else
      parts.push_back(eleTy.getPointerTo());
    parts.push_back(getDescFieldTypeModel<1>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<2>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<3>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<4>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<5>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<6>()(&getContext()));
    if (rank == unknownRank()) {
      if (auto seqTy = ele.dyn_cast<SequenceType>()) {
        if (seqTy.getEleTy().isa<CharacterType>()) {
          auto dim = seqTy.getDimension();
          if (dim == 0) {
            emitError(UnknownLoc::get(&getContext()))
                << "missing length in character sequence type :" << eleTy;
            rank = 0;
          } else {
            rank = dim - 1;
          }
        } else {
          rank = seqTy.getDimension();
        }
      } else {
        rank = 0;
      }
    }
    if (rank > 0) {
      auto rowTy = getDescFieldTypeModel<7>()(&getContext());
      parts.push_back(mlir::LLVM::LLVMType::getArrayTy(rowTy, rank));
    }
    // opt-type-ptr: i8* (see fir.tdesc)
    if (requiresExtendedDesc()) {
      parts.push_back(getExtendedDescFieldTypeModel<8>()(&getContext()));
      parts.push_back(getExtendedDescFieldTypeModel<9>()(&getContext()));
      auto rowTy = getExtendedDescFieldTypeModel<10>()(&getContext());
      unsigned numLenParams = 0; // FIXME
      parts.push_back(mlir::LLVM::LLVMType::getArrayTy(rowTy, numLenParams));
    }
    return mlir::LLVM::LLVMType::getStructTy(&getContext(), parts)
        .getPointerTo();
  }

  // fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
  mlir::LLVM::LLVMType convertBoxProcType(BoxProcType boxproc) {
    auto funcTy = convertType(boxproc.getEleTy());
    auto ptrTy = unwrap(funcTy).getPointerTo();
    auto i8Ty = mlir::LLVM::LLVMType::getInt8Ty(&getContext());
    SmallVector<mlir::LLVM::LLVMType, 2> tuple{ptrTy, i8Ty};
    return mlir::LLVM::LLVMType::getStructTy(&getContext(), tuple);
  }

  unsigned characterBitsize(fir::CharacterType charTy) {
    return kindMapping.getCharacterBitsize(charTy.getFKind());
  }

  // fir.char<n>  -->  llvm<"ix*">   where ix is scaled by kind mapping
  mlir::LLVM::LLVMType convertCharType(fir::CharacterType charTy) {
    return mlir::LLVM::LLVMType::getIntNTy(&getContext(),
                                           characterBitsize(charTy));
  }

  // Convert a complex value's element type based on its Fortran kind.
  mlir::LLVM::LLVMType convertComplexPartType(fir::KindTy kind) {
    auto realID = kindMapping.getComplexTypeID(kind);
    return fromRealTypeID(realID, kind);
  }

  // Use the target specifics to figure out how to map complex to LLVM IR. The
  // use of complex values in function signatures is handled before conversion
  // to LLVM IR dialect here.
  //
  // fir.complex<T> | std.complex<T>    --> llvm<"{t,t}">
  template <typename C>
  mlir::LLVM::LLVMType convertComplexType(C cmplx) {
    LLVM_DEBUG(llvm::dbgs() << "type convert: " << cmplx << '\n');
    auto eleTy = cmplx.getElementType();
    return unwrap(convertType(specifics->complexMemoryType(eleTy)));
  }

  // Get the default size of INTEGER. (The default size might have been set on
  // the command line.)
  mlir::LLVM::LLVMType getDefaultInt() {
    return mlir::LLVM::LLVMType::getIntNTy(
        &getContext(),
        kindMapping.getIntegerBitsize(kindMapping.defaultIntegerKind()));
  }

  template <typename A>
  mlir::LLVM::LLVMType convertPointerLike(A &ty) {
    mlir::Type eleTy = ty.getEleTy();
    // A sequence type is a special case. A sequence of runtime size on its
    // interior dimensions lowers to a memory reference. In that case, we
    // degenerate the array and do not want a the type to become `T**` but
    // merely `T*`.
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (!seqTy.hasConstantShape()) {
        if (seqTy.hasConstantInterior())
          return unwrap(convertType(seqTy));
        eleTy = seqTy.getEleTy();
      }
    }
    return unwrap(convertType(eleTy)).getPointerTo();
  }

  // convert a front-end kind value to either a std or LLVM IR dialect type
  // fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
  mlir::LLVM::LLVMType convertRealType(fir::KindTy kind) {
    return fromRealTypeID(kindMapping.getRealTypeID(kind), kind);
  }

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  mlir::LLVM::LLVMType convertRecordType(fir::RecordType derived) {
    auto name = derived.getName();
    // The cache is needed to keep a unique mapping from name -> StructType
    auto iter = identStructCache.find(name);
    if (iter != identStructCache.end())
      return iter->second;
    auto st = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);
    identStructCache[name] = st;
    SmallVector<mlir::LLVM::LLVMType, 8> members;
    for (auto mem : derived.getTypeList())
      members.push_back(convertType(mem.second).cast<mlir::LLVM::LLVMType>());
    mlir::LLVM::LLVMType::setStructTyBody(st, members);
    return st;
  }

  // fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
  mlir::LLVM::LLVMType convertSequenceType(SequenceType seq) {
    auto baseTy = unwrap(convertType(seq.getEleTy()));
    auto shape = seq.getShape();
    auto constRows = seq.getConstantRows();
    if (constRows) {
      decltype(constRows) i = constRows;
      for (auto e : shape) {
        baseTy = mlir::LLVM::LLVMType::getArrayTy(baseTy, e);
        if (--i == 0)
          break;
      }
      if (seq.hasConstantShape())
        return baseTy;
    }
    return baseTy.getPointerTo();
  }

  // fir.tdesc<any>  -->  llvm<"i8*">
  // FIXME: for now use a void*, however pointer identity is not sufficient for
  // the f18 object v. class distinction
  mlir::LLVM::LLVMType convertTypeDescType(mlir::MLIRContext *ctx) {
    return mlir::LLVM::LLVMType::getInt8PtrTy(&getContext());
  }

  /// Convert llvm::Type::TypeID to mlir::LLVM::LLVMType
  mlir::LLVM::LLVMType fromRealTypeID(llvm::Type::TypeID typeID,
                                      fir::KindTy kind) {
    switch (typeID) {
    case llvm::Type::TypeID::HalfTyID:
      return mlir::LLVM::LLVMType::getHalfTy(&getContext());
    case llvm::Type::TypeID::FloatTyID:
      return mlir::LLVM::LLVMType::getFloatTy(&getContext());
    case llvm::Type::TypeID::DoubleTyID:
      return mlir::LLVM::LLVMType::getDoubleTy(&getContext());
    case llvm::Type::TypeID::X86_FP80TyID:
      return mlir::LLVM::LLVMType::getX86_FP80Ty(&getContext());
    case llvm::Type::TypeID::FP128TyID:
      return mlir::LLVM::LLVMType::getFP128Ty(&getContext());
    default:
      emitError(UnknownLoc::get(&getContext()))
          << "unsupported type: !fir.real<" << kind << ">";
      return {};
    }
  }

  /// HACK: cloned from LLVMTypeConverter since this is private there
  mlir::LLVM::LLVMType unwrap(mlir::Type type) {
    if (!type)
      return nullptr;
    auto *mlirContext = type.getContext();
    auto wrappedLLVMType = type.dyn_cast<mlir::LLVM::LLVMType>();
    if (!wrappedLLVMType)
      emitError(UnknownLoc::get(mlirContext),
                "conversion resulted in a non-LLVM type");
    return wrappedLLVMType;
  }

  /// Returns false iff the sequence type has a shape and the shape is constant.
  static bool unknownShape(SequenceType::Shape shape) {
    // does the shape even exist?
    auto size = shape.size();
    if (size == 0)
      return true;
    // if it exists, are any dimensions deferred?
    for (decltype(size) i = 0, sz = size; i < sz; ++i)
      if (shape[i] == SequenceType::getUnknownExtent())
        return true;
    return false;
  }

  /// Does this record type have dynamically inlined subobjects? Note: this
  /// should not look through references as they are not inlined.
  static bool dynamicallySized(fir::RecordType seqTy) {
    for (auto field : seqTy.getTypeList()) {
      if (auto arr = field.second.dyn_cast<SequenceType>()) {
        if (unknownShape(arr.getShape()))
          return true;
      } else if (auto rec = field.second.dyn_cast<fir::RecordType>()) {
        if (dynamicallySized(rec))
          return true;
      }
    }
    return false;
  }

  static bool dynamicallySized(mlir::Type ty) {
    if (auto arr = ty.dyn_cast<SequenceType>())
      ty = arr.getEleTy();
    if (auto rec = ty.dyn_cast<fir::RecordType>())
      return dynamicallySized(rec);
    return false;
  }

  NameUniquer &getUniquer() { return uniquer; }

  KindMapping &getKindMap() { return kindMapping; }

private:
  KindMapping kindMapping;
  NameUniquer &uniquer;
  std::unique_ptr<CodeGenSpecifics> specifics;
  static StringMap<mlir::LLVM::LLVMType> identStructCache;
};

} // namespace fir

#endif // OPTIMIZER_CODEGEN_TYPECONVERTER_H
