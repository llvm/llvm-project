//===-- TypeConverter.h -- type conversion ----------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H
#define FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H

namespace fir {

/// FIR type converter
/// This converts FIR types to LLVM types (for now)
class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LLVMTypeConverter(mlir::ModuleOp module)
      : mlir::LLVMTypeConverter(module.getContext()),
        kindMapping(getKindMapping(module)),
        specifics(CodeGenSpecifics::get(module.getContext(),
                                        getTargetTriple(module),
                                        getKindMapping(module))) {
    LLVM_DEBUG(llvm::dbgs() << "FIR type converter\n");

    // Each conversion should return a value of type mlir::Type.
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
      return mlir::IntegerType::get(field.getContext(), 32);
    });
    addConversion([&](HeapType heap) { return convertPointerLike(heap); });
    addConversion([&](fir::IntegerType intTy) {
      return mlir::IntegerType::get(
          &getContext(), kindMapping.getIntegerBitsize(intTy.getFKind()));
    });
    addConversion([&](LenType field) {
      return mlir::IntegerType::get(field.getContext(), 32);
    });
    addConversion([&](fir::LogicalType boolTy) {
      return mlir::IntegerType::get(
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
      return mlir::VectorType::get(llvm::ArrayRef<int64_t>(vecTy.getLen()),
                                   unwrap(convertType(vecTy.getEleTy())));
    });
    addConversion([&](mlir::TupleType tuple) {
      LLVM_DEBUG(llvm::dbgs() << "type convert: " << tuple << '\n');
      llvm::SmallVector<mlir::Type> inMembers;
      tuple.getFlattenedTypes(inMembers);
      llvm::SmallVector<mlir::Type> members;
      for (auto mem : inMembers) {
        // Prevent fir.box from degenerating to a pointer to a descriptor in the
        // context of a tuple type.
        if (auto box = mem.dyn_cast<fir::BoxType>())
          members.push_back(convertBoxTypeAsStruct(box));
        else
          members.push_back(convertType(mem).cast<mlir::Type>());
      }
      return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), members,
                                                    /*isPacked=*/false);
    });
    addConversion([&](mlir::NoneType none) {
      return mlir::LLVM::LLVMStructType::getLiteral(
          none.getContext(), llvm::None, /*isPacked=*/false);
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
  mlir::Type offsetType() { return mlir::IntegerType::get(&getContext(), 32); }

  // i64 can be used to index into aggregates like arrays
  mlir::Type indexType() { return mlir::IntegerType::get(&getContext(), 64); }

  // TODO
  bool requiresExtendedDesc() { return false; }

  // Magic value to indicate we do not know the rank of an entity, either
  // because it is assumed rank or because we have not determined it yet.
  static constexpr int unknownRank() { return -1; }
  // This corresponds to the descriptor as defined ISO_Fortran_binding.h and the
  // addendum defined in descriptor.h.
  mlir::Type convertBoxType(BoxType box, int rank = unknownRank()) {
    // (buffer*, ele-size, rank, type-descriptor, attribute, [dims])
    SmallVector<mlir::Type> parts;
    mlir::Type ele = box.getEleTy();
    // remove fir.heap/fir.ref/fir.ptr
    if (auto removeIndirection = fir::dyn_cast_ptrEleTy(ele))
      ele = removeIndirection;
    auto eleTy = unwrap(convertType(ele));
    // buffer*
    if (ele.isa<SequenceType>() && eleTy.isa<mlir::LLVM::LLVMPointerType>())
      parts.push_back(eleTy);
    else
      parts.push_back(mlir::LLVM::LLVMPointerType::get(eleTy));
    parts.push_back(getDescFieldTypeModel<1>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<2>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<3>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<4>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<5>()(&getContext()));
    parts.push_back(getDescFieldTypeModel<6>()(&getContext()));
    if (rank == unknownRank()) {
      if (auto seqTy = ele.dyn_cast<SequenceType>())
        rank = seqTy.getDimension();
      else
        rank = 0;
    }
    if (rank > 0) {
      auto rowTy = getDescFieldTypeModel<7>()(&getContext());
      parts.push_back(mlir::LLVM::LLVMArrayType::get(rowTy, rank));
    }
    // opt-type-ptr: i8* (see fir.tdesc)
    if (requiresExtendedDesc()) {
      parts.push_back(getExtendedDescFieldTypeModel<8>()(&getContext()));
      parts.push_back(getExtendedDescFieldTypeModel<9>()(&getContext()));
      auto rowTy = getExtendedDescFieldTypeModel<10>()(&getContext());
      unsigned numLenParams = 0; // FIXME
      parts.push_back(mlir::LLVM::LLVMArrayType::get(rowTy, numLenParams));
      TODO_NOLOC("extended descriptor");
    }
    return mlir::LLVM::LLVMPointerType::get(
        mlir::LLVM::LLVMStructType::getLiteral(&getContext(), parts,
                                               /*isPacked=*/false));
  }
  /// Convert fir.box type to the corresponding llvm struct type instead of a
  /// pointer to this struct type.
  mlir::Type convertBoxTypeAsStruct(BoxType box) {
    return convertBoxType(box)
        .cast<mlir::LLVM::LLVMPointerType>()
        .getElementType();
  }

  // fir.boxproc<any>  -->  llvm<"{ any*, i8* }">
  mlir::Type convertBoxProcType(BoxProcType boxproc) {
    auto funcTy = convertType(boxproc.getEleTy());
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(unwrap(funcTy));
    auto i8PtrTy = mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));
    llvm::SmallVector<mlir::Type, 2> tuple = {ptrTy, i8PtrTy};
    return mlir::LLVM::LLVMStructType::getLiteral(&getContext(), tuple,
                                                  /*isPacked=*/false);
  }

  unsigned characterBitsize(fir::CharacterType charTy) {
    return kindMapping.getCharacterBitsize(charTy.getFKind());
  }

  // fir.char<n>  -->  llvm<"ix*">   where ix is scaled by kind mapping
  mlir::Type convertCharType(fir::CharacterType charTy) {
    auto iTy = mlir::IntegerType::get(&getContext(), characterBitsize(charTy));
    if (charTy.getLen() == fir::CharacterType::unknownLen())
      return iTy;
    return mlir::LLVM::LLVMArrayType::get(iTy, charTy.getLen());
  }

  // Convert a complex value's element type based on its Fortran kind.
  mlir::Type convertComplexPartType(fir::KindTy kind) {
    auto realID = kindMapping.getComplexTypeID(kind);
    return fromRealTypeID(realID, kind);
  }

  // Use the target specifics to figure out how to map complex to LLVM IR. The
  // use of complex values in function signatures is handled before conversion
  // to LLVM IR dialect here.
  //
  // fir.complex<T> | std.complex<T>    --> llvm<"{t,t}">
  template <typename C>
  mlir::Type convertComplexType(C cmplx) {
    LLVM_DEBUG(llvm::dbgs() << "type convert: " << cmplx << '\n');
    auto eleTy = cmplx.getElementType();
    return unwrap(convertType(specifics->complexMemoryType(eleTy)));
  }

  // Get the default size of INTEGER. (The default size might have been set on
  // the command line.)
  mlir::Type getDefaultInt() {
    return mlir::IntegerType::get(
        &getContext(),
        kindMapping.getIntegerBitsize(kindMapping.defaultIntegerKind()));
  }

  template <typename A>
  mlir::Type convertPointerLike(A &ty) {
    mlir::Type eleTy = ty.getEleTy();
    // A sequence type is a special case. A sequence of runtime size on its
    // interior dimensions lowers to a memory reference. In that case, we
    // degenerate the array and do not want a the type to become `T**` but
    // merely `T*`.
    if (auto seqTy = eleTy.dyn_cast<fir::SequenceType>()) {
      if (!seqTy.hasConstantShape() ||
          characterWithDynamicLen(seqTy.getEleTy())) {
        if (seqTy.hasConstantInterior())
          return unwrap(convertType(seqTy));
        eleTy = seqTy.getEleTy();
      }
    }
    // fir.ref<fir.box> is a special case because fir.box type is already
    // a pointer to a Fortran descriptor at the LLVM IR level. This implies
    // that a fir.ref<fir.box>, that is the address of fir.box is actually
    // the same as a fir.box at the LLVM level.
    // The distinction is kept in fir to denote when a descriptor is expected
    // to be mutable (fir.ref<fir.box>) and when it is not (fir.box).
    if (eleTy.isa<fir::BoxType>())
      return unwrap(convertType(eleTy));

    return mlir::LLVM::LLVMPointerType::get(unwrap(convertType(eleTy)));
  }

  // convert a front-end kind value to either a std or LLVM IR dialect type
  // fir.real<n>  -->  llvm.anyfloat  where anyfloat is a kind mapping
  mlir::Type convertRealType(fir::KindTy kind) {
    return fromRealTypeID(kindMapping.getRealTypeID(kind), kind);
  }

  // fir.type<name(p : TY'...){f : TY...}>  -->  llvm<"%name = { ty... }">
  mlir::Type convertRecordType(fir::RecordType derived) {
    auto name = derived.getName();
    // The cache is needed to keep a unique mapping from name -> StructType
    auto iter = identStructCache.find(name);
    if (iter != identStructCache.end())
      return iter->second;
    auto st = mlir::LLVM::LLVMStructType::getIdentified(&getContext(), name);
    identStructCache[name] = st;
    llvm::SmallVector<mlir::Type> members;
    for (auto mem : derived.getTypeList()) {
      // Prevent fir.box from degenerating to a pointer to a descriptor in the
      // context of a record type.
      if (auto box = mem.second.dyn_cast<fir::BoxType>())
        members.push_back(convertBoxTypeAsStruct(box));
      else
        members.push_back(convertType(mem.second).cast<mlir::Type>());
    }
    (void)st.setBody(members, /*isPacked=*/false);
    return st;
  }

  // fir.array<c ... :any>  -->  llvm<"[...[c x any]]">
  mlir::Type convertSequenceType(SequenceType seq) {
    auto baseTy = unwrap(convertType(seq.getEleTy()));
    if (characterWithDynamicLen(seq.getEleTy()))
      return mlir::LLVM::LLVMPointerType::get(baseTy);
    auto shape = seq.getShape();
    auto constRows = seq.getConstantRows();
    if (constRows) {
      decltype(constRows) i = constRows;
      for (auto e : shape) {
        baseTy = mlir::LLVM::LLVMArrayType::get(baseTy, e);
        if (--i == 0)
          break;
      }
      if (seq.hasConstantShape())
        return baseTy;
    }
    return mlir::LLVM::LLVMPointerType::get(baseTy);
  }

  // fir.tdesc<any>  -->  llvm<"i8*">
  // FIXME: for now use a void*, however pointer identity is not sufficient for
  // the f18 object v. class distinction
  mlir::Type convertTypeDescType(mlir::MLIRContext *ctx) {
    return mlir::LLVM::LLVMPointerType::get(
        mlir::IntegerType::get(&getContext(), 8));
  }

  /// Convert llvm::Type::TypeID to mlir::Type
  mlir::Type fromRealTypeID(llvm::Type::TypeID typeID, fir::KindTy kind) {
    switch (typeID) {
    case llvm::Type::TypeID::HalfTyID:
      return mlir::FloatType::getF16(&getContext());
    case llvm::Type::TypeID::FloatTyID:
      return mlir::FloatType::getF32(&getContext());
    case llvm::Type::TypeID::DoubleTyID:
      return mlir::FloatType::getF64(&getContext());
    case llvm::Type::TypeID::X86_FP80TyID:
      return mlir::FloatType::getF80(&getContext());
    case llvm::Type::TypeID::FP128TyID:
      return mlir::FloatType::getF128(&getContext());
    default:
      emitError(UnknownLoc::get(&getContext()))
          << "unsupported type: !fir.real<" << kind << ">";
      return {};
    }
  }

  /// HACK: cloned from LLVMTypeConverter since this is private there
  mlir::Type unwrap(mlir::Type type) {
    if (!type)
      return nullptr;
    auto *mlirContext = type.getContext();
    auto wrappedLLVMType = type.dyn_cast<mlir::Type>();
    if (!wrappedLLVMType)
      emitError(UnknownLoc::get(mlirContext),
                "conversion resulted in a non-LLVM type");
    return wrappedLLVMType;
  }

  KindMapping &getKindMap() { return kindMapping; }

private:
  KindMapping kindMapping;
  std::unique_ptr<CodeGenSpecifics> specifics;
  static StringMap<mlir::Type> identStructCache;
};

} // namespace fir

#endif // FORTRAN_OPTIMIZER_CODEGEN_TYPECONVERTER_H
