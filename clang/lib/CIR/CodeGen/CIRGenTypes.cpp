#include "CIRGenTypes.h"

#include "CIRGenFunctionInfo.h"
#include "CIRGenModule.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"

#include <cassert>

using namespace clang;
using namespace clang::CIRGen;

CIRGenTypes::CIRGenTypes(CIRGenModule &genModule)
    : cgm(genModule), astContext(genModule.getASTContext()),
      builder(cgm.getBuilder()),
      theABIInfo(cgm.getTargetCIRGenInfo().getABIInfo()) {}

CIRGenTypes::~CIRGenTypes() {
  for (auto i = functionInfos.begin(), e = functionInfos.end(); i != e;)
    delete &*i++;
}

mlir::MLIRContext &CIRGenTypes::getMLIRContext() const {
  return *builder.getContext();
}

/// Return true if the specified type in a function parameter or result position
/// can be converted to a CIR type at this point. This boils down to being
/// whether it is complete, as well as whether we've temporarily deferred
/// expanding the type because we're in a recursive context.
bool CIRGenTypes::isFuncParamTypeConvertible(clang::QualType type) {
  // Some ABIs cannot have their member pointers represented in LLVM IR unless
  // certain circumstances have been reached.
  assert(!type->getAs<MemberPointerType>() && "NYI");

  // If this isn't a tag type, we can convert it.
  const TagType *tagType = type->getAs<TagType>();
  if (!tagType)
    return true;

  // Function types involving incomplete class types are problematic in MLIR.
  return !tagType->isIncompleteType();
}

/// Code to verify a given function type is complete, i.e. the return type and
/// all of the parameter types are complete. Also check to see if we are in a
/// RS_StructPointer context, and if so whether any struct types have been
/// pended. If so, we don't want to ask the ABI lowering code to handle a type
/// that cannot be converted to a CIR type.
bool CIRGenTypes::isFuncTypeConvertible(const FunctionType *ft) {
  if (!isFuncParamTypeConvertible(ft->getReturnType()))
    return false;

  if (const auto *fpt = dyn_cast<FunctionProtoType>(ft))
    for (unsigned i = 0, e = fpt->getNumParams(); i != e; i++)
      if (!isFuncParamTypeConvertible(fpt->getParamType(i)))
        return false;

  return true;
}

mlir::Type CIRGenTypes::convertFunctionTypeInternal(QualType qft) {
  assert(qft.isCanonical());
  const FunctionType *ft = cast<FunctionType>(qft.getTypePtr());
  // First, check whether we can build the full function type. If the function
  // type depends on an incomplete type (e.g. a struct or enum), we cannot lower
  // the function type.
  if (!isFuncTypeConvertible(ft)) {
    cgm.errorNYI(SourceLocation(), "function type involving an incomplete type",
                 qft);
    return cir::FuncType::get(SmallVector<mlir::Type, 1>{}, cgm.VoidTy);
  }

  const CIRGenFunctionInfo *fi;
  if (const auto *fpt = dyn_cast<FunctionProtoType>(ft)) {
    fi = &arrangeFreeFunctionType(
        CanQual<FunctionProtoType>::CreateUnsafe(QualType(fpt, 0)));
  } else {
    const FunctionNoProtoType *fnpt = cast<FunctionNoProtoType>(ft);
    fi = &arrangeFreeFunctionType(
        CanQual<FunctionNoProtoType>::CreateUnsafe(QualType(fnpt, 0)));
  }

  mlir::Type resultType = getFunctionType(*fi);

  return resultType;
}

// This is CIR's version of CodeGenTypes::addRecordTypeName. It isn't shareable
// because CIR has different uniquing requirements.
std::string CIRGenTypes::getRecordTypeName(const clang::RecordDecl *recordDecl,
                                           StringRef suffix) {
  llvm::SmallString<256> typeName;
  llvm::raw_svector_ostream outStream(typeName);

  PrintingPolicy policy = recordDecl->getASTContext().getPrintingPolicy();
  policy.SuppressInlineNamespace = false;
  policy.AlwaysIncludeTypeForTemplateArgument = true;
  policy.PrintAsCanonical = true;
  policy.SuppressTagKeyword = true;

  if (recordDecl->getIdentifier())
    astContext.getRecordType(recordDecl).print(outStream, policy);
  else if (auto *typedefNameDecl = recordDecl->getTypedefNameForAnonDecl())
    typedefNameDecl->printQualifiedName(outStream, policy);
  else
    outStream << builder.getUniqueAnonRecordName();

  if (!suffix.empty())
    outStream << suffix;

  return builder.getUniqueRecordName(std::string(typeName));
}

/// Return true if the specified type is already completely laid out.
bool CIRGenTypes::isRecordLayoutComplete(const Type *ty) const {
  const auto it = recordDeclTypes.find(ty);
  return it != recordDeclTypes.end() && it->second.isComplete();
}

// We have multiple forms of this function that call each other, so we need to
// declare one in advance.
static bool
isSafeToConvert(QualType qt, CIRGenTypes &cgt,
                llvm::SmallPtrSetImpl<const RecordDecl *> &alreadyChecked);

/// Return true if it is safe to convert the specified record decl to CIR and
/// lay it out, false if doing so would cause us to get into a recursive
/// compilation mess.
static bool
isSafeToConvert(const RecordDecl *rd, CIRGenTypes &cgt,
                llvm::SmallPtrSetImpl<const RecordDecl *> &alreadyChecked) {
  // If we have already checked this type (maybe the same type is used by-value
  // multiple times in multiple record fields, don't check again.
  if (!alreadyChecked.insert(rd).second)
    return true;

  const Type *key = cgt.getASTContext().getTagDeclType(rd).getTypePtr();

  // If this type is already laid out, converting it is a noop.
  if (cgt.isRecordLayoutComplete(key))
    return true;

  // If this type is currently being laid out, we can't recursively compile it.
  if (cgt.isRecordBeingLaidOut(key))
    return false;

  // If this type would require laying out bases that are currently being laid
  // out, don't do it.  This includes virtual base classes which get laid out
  // when a class is translated, even though they aren't embedded by-value into
  // the class.
  if (isa<CXXRecordDecl>(rd)) {
    assert(!cir::MissingFeatures::cxxSupport());
    cgt.getCGModule().errorNYI(rd->getSourceRange(),
                               "isSafeToConvert: CXXRecordDecl");
    return false;
  }

  // If this type would require laying out members that are currently being laid
  // out, don't do it.
  for (const FieldDecl *field : rd->fields())
    if (!isSafeToConvert(field->getType(), cgt, alreadyChecked))
      return false;

  // If there are no problems, lets do it.
  return true;
}

/// Return true if it is safe to convert this field type, which requires the
/// record elements contained by-value to all be recursively safe to convert.
static bool
isSafeToConvert(QualType qt, CIRGenTypes &cgt,
                llvm::SmallPtrSetImpl<const RecordDecl *> &alreadyChecked) {
  // Strip off atomic type sugar.
  if (const auto *at = qt->getAs<AtomicType>())
    qt = at->getValueType();

  // If this is a record, check it.
  if (const auto *rt = qt->getAs<RecordType>())
    return isSafeToConvert(rt->getDecl(), cgt, alreadyChecked);

  // If this is an array, check the elements, which are embedded inline.
  if (const auto *at = cgt.getASTContext().getAsArrayType(qt))
    return isSafeToConvert(at->getElementType(), cgt, alreadyChecked);

  // Otherwise, there is no concern about transforming this. We only care about
  // things that are contained by-value in a record that can have another
  // record as a member.
  return true;
}

// Return true if it is safe to convert the specified record decl to CIR and lay
// it out, false if doing so would cause us to get into a recursive compilation
// mess.
static bool isSafeToConvert(const RecordDecl *rd, CIRGenTypes &cgt) {
  // If no records are being laid out, we can certainly do this one.
  if (cgt.noRecordsBeingLaidOut())
    return true;

  llvm::SmallPtrSet<const RecordDecl *, 16> alreadyChecked;
  return isSafeToConvert(rd, cgt, alreadyChecked);
}

/// Lay out a tagged decl type like struct or union.
mlir::Type CIRGenTypes::convertRecordDeclType(const clang::RecordDecl *rd) {
  // TagDecl's are not necessarily unique, instead use the (clang) type
  // connected to the decl.
  const Type *key = astContext.getTagDeclType(rd).getTypePtr();
  cir::RecordType entry = recordDeclTypes[key];

  // If we don't have an entry for this record yet, create one.
  // We create an incomplete type initially. If `rd` is complete, we will
  // add the members below.
  if (!entry) {
    auto name = getRecordTypeName(rd, "");
    entry = builder.getIncompleteRecordTy(name, rd);
    recordDeclTypes[key] = entry;
  }

  rd = rd->getDefinition();
  if (!rd || !rd->isCompleteDefinition() || entry.isComplete())
    return entry;

  // If converting this type would cause us to infinitely loop, don't do it!
  if (!isSafeToConvert(rd, *this)) {
    cgm.errorNYI(rd->getSourceRange(), "recursive record layout");
    return entry;
  }

  // Okay, this is a definition of a type. Compile the implementation now.
  bool insertResult = recordsBeingLaidOut.insert(key).second;
  (void)insertResult;
  assert(insertResult && "isSafeToCovert() should have caught this.");

  // Force conversion of non-virtual base classes recursively.
  if (const auto *cxxRecordDecl = dyn_cast<CXXRecordDecl>(rd)) {
    if (cxxRecordDecl->getNumBases() > 0) {
      cgm.errorNYI(rd->getSourceRange(),
                   "convertRecordDeclType: derived CXXRecordDecl");
    }
  }

  // Layout fields.
  std::unique_ptr<CIRGenRecordLayout> layout = computeRecordLayout(rd, &entry);
  recordDeclTypes[key] = entry;
  cirGenRecordLayouts[key] = std::move(layout);

  // We're done laying out this record.
  bool eraseResult = recordsBeingLaidOut.erase(key);
  (void)eraseResult;
  assert(eraseResult && "record not in RecordsBeingLaidOut set?");

  // If this record blocked a FunctionType conversion, then recompute whatever
  // was derived from that.
  assert(!cir::MissingFeatures::skippedLayout());

  // If we're done converting the outer-most record, then convert any deferred
  // records as well.
  assert(!cir::MissingFeatures::recursiveRecordLayout());

  return entry;
}

mlir::Type CIRGenTypes::convertType(QualType type) {
  type = astContext.getCanonicalType(type);
  const Type *ty = type.getTypePtr();

  // Process record types before the type cache lookup.
  if (const auto *recordType = dyn_cast<RecordType>(type))
    return convertRecordDeclType(recordType->getDecl());

  // Has the type already been processed?
  TypeCacheTy::iterator tci = typeCache.find(ty);
  if (tci != typeCache.end())
    return tci->second;

  // For types that haven't been implemented yet or are otherwise unsupported,
  // report an error and return 'int'.

  mlir::Type resultType = nullptr;
  switch (ty->getTypeClass()) {
  case Type::Record:
    llvm_unreachable("Should have been handled above");

  case Type::Builtin: {
    switch (cast<BuiltinType>(ty)->getKind()) {
    // void
    case BuiltinType::Void:
      resultType = cgm.VoidTy;
      break;

    // bool
    case BuiltinType::Bool:
      resultType = cir::BoolType::get(&getMLIRContext());
      break;

    // Signed integral types.
    case BuiltinType::Char_S:
    case BuiltinType::Int:
    case BuiltinType::Int128:
    case BuiltinType::Long:
    case BuiltinType::LongLong:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::WChar_S:
      resultType =
          cir::IntType::get(&getMLIRContext(), astContext.getTypeSize(ty),
                            /*isSigned=*/true);
      break;
    // Unsigned integral types.
    case BuiltinType::Char8:
    case BuiltinType::Char16:
    case BuiltinType::Char32:
    case BuiltinType::Char_U:
    case BuiltinType::UChar:
    case BuiltinType::UInt:
    case BuiltinType::UInt128:
    case BuiltinType::ULong:
    case BuiltinType::ULongLong:
    case BuiltinType::UShort:
    case BuiltinType::WChar_U:
      resultType =
          cir::IntType::get(&getMLIRContext(), astContext.getTypeSize(ty),
                            /*isSigned=*/false);
      break;

    // Floating-point types
    case BuiltinType::Float16:
      resultType = cgm.FP16Ty;
      break;
    case BuiltinType::Half:
      if (astContext.getLangOpts().NativeHalfType ||
          !astContext.getTargetInfo().useFP16ConversionIntrinsics()) {
        resultType = cgm.FP16Ty;
      } else {
        cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
        resultType = cgm.SInt32Ty;
      }
      break;
    case BuiltinType::BFloat16:
      resultType = cgm.BFloat16Ty;
      break;
    case BuiltinType::Float:
      assert(&astContext.getFloatTypeSemantics(type) ==
                 &llvm::APFloat::IEEEsingle() &&
             "ClangIR NYI: 'float' in a format other than IEEE 32-bit");
      resultType = cgm.FloatTy;
      break;
    case BuiltinType::Double:
      assert(&astContext.getFloatTypeSemantics(type) ==
                 &llvm::APFloat::IEEEdouble() &&
             "ClangIR NYI: 'double' in a format other than IEEE 64-bit");
      resultType = cgm.DoubleTy;
      break;
    case BuiltinType::LongDouble:
      resultType =
          builder.getLongDoubleTy(astContext.getFloatTypeSemantics(type));
      break;
    case BuiltinType::Float128:
      resultType = cgm.FP128Ty;
      break;
    case BuiltinType::Ibm128:
      cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
      resultType = cgm.SInt32Ty;
      break;

    case BuiltinType::NullPtr:
      // Add proper CIR type for it? this looks mostly useful for sema related
      // things (like for overloads accepting void), for now, given that
      // `sizeof(std::nullptr_t)` is equal to `sizeof(void *)`, model
      // std::nullptr_t as !cir.ptr<!void>
      resultType = builder.getVoidPtrTy();
      break;

    default:
      cgm.errorNYI(SourceLocation(), "processing of built-in type", type);
      resultType = cgm.SInt32Ty;
      break;
    }
    break;
  }

  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType *refTy = cast<ReferenceType>(ty);
    QualType elemTy = refTy->getPointeeType();
    auto pointeeType = convertTypeForMem(elemTy);
    resultType = builder.getPointerTo(pointeeType);
    assert(resultType && "Cannot get pointer type?");
    break;
  }

  case Type::Pointer: {
    const PointerType *ptrTy = cast<PointerType>(ty);
    QualType elemTy = ptrTy->getPointeeType();
    assert(!elemTy->isConstantMatrixType() && "not implemented");

    mlir::Type pointeeType = convertType(elemTy);

    resultType = builder.getPointerTo(pointeeType);
    break;
  }

  case Type::ConstantArray: {
    const ConstantArrayType *arrTy = cast<ConstantArrayType>(ty);
    mlir::Type elemTy = convertTypeForMem(arrTy->getElementType());
    resultType = cir::ArrayType::get(elemTy, arrTy->getSize().getZExtValue());
    break;
  }

  case Type::ExtVector:
  case Type::Vector: {
    const VectorType *vec = cast<VectorType>(ty);
    const mlir::Type elemTy = convertType(vec->getElementType());
    resultType = cir::VectorType::get(elemTy, vec->getNumElements());
    break;
  }

  case Type::Enum: {
    // TODO(cir): Implement updateCompletedType for enums.
    assert(!cir::MissingFeatures::updateCompletedType());
    const EnumDecl *ED = cast<EnumType>(ty)->getDecl();
    if (auto integerType = ED->getIntegerType(); !integerType.isNull())
      return convertType(integerType);
    // Return a placeholder 'i32' type.  This can be changed later when the
    // type is defined (see UpdateCompletedType), but is likely to be the
    // "right" answer.
    resultType = cgm.UInt32Ty;
    break;
  }

  case Type::FunctionNoProto:
  case Type::FunctionProto:
    resultType = convertFunctionTypeInternal(type);
    break;

  case Type::BitInt: {
    const auto *bitIntTy = cast<BitIntType>(type);
    if (bitIntTy->getNumBits() > cir::IntType::maxBitwidth()) {
      cgm.errorNYI(SourceLocation(), "large _BitInt type", type);
      resultType = cgm.SInt32Ty;
    } else {
      resultType = cir::IntType::get(&getMLIRContext(), bitIntTy->getNumBits(),
                                     bitIntTy->isSigned());
    }
    break;
  }

  default:
    cgm.errorNYI(SourceLocation(), "processing of type",
                 type->getTypeClassName());
    resultType = cgm.SInt32Ty;
    break;
  }

  assert(resultType && "Type conversion not yet implemented");

  typeCache[ty] = resultType;
  return resultType;
}

mlir::Type CIRGenTypes::convertTypeForMem(clang::QualType qualType,
                                          bool forBitField) {
  assert(!qualType->isConstantMatrixType() && "Matrix types NYI");

  mlir::Type convertedType = convertType(qualType);

  assert(!forBitField && "Bit fields NYI");

  // If this is a bit-precise integer type in a bitfield representation, map
  // this integer to the target-specified size.
  if (forBitField && qualType->isBitIntType())
    assert(!qualType->isBitIntType() && "Bit field with type _BitInt NYI");

  return convertedType;
}

/// Return record layout info for the given record decl.
const CIRGenRecordLayout &
CIRGenTypes::getCIRGenRecordLayout(const RecordDecl *rd) {
  const auto *key = astContext.getTagDeclType(rd).getTypePtr();

  // If we have already computed the layout, return it.
  auto it = cirGenRecordLayouts.find(key);
  if (it != cirGenRecordLayouts.end())
    return *it->second;

  // Compute the type information.
  convertRecordDeclType(rd);

  // Now try again.
  it = cirGenRecordLayouts.find(key);

  assert(it != cirGenRecordLayouts.end() &&
         "Unable to find record layout information for type");
  return *it->second;
}

bool CIRGenTypes::isZeroInitializable(clang::QualType t) {
  if (t->getAs<PointerType>())
    return astContext.getTargetNullPointerValue(t) == 0;

  if (const auto *at = astContext.getAsArrayType(t)) {
    if (isa<IncompleteArrayType>(at))
      return true;

    if (const auto *cat = dyn_cast<ConstantArrayType>(at))
      if (astContext.getConstantArrayElementCount(cat) == 0)
        return true;
  }

  if (const RecordType *rt = t->getAs<RecordType>()) {
    const RecordDecl *rd = rt->getDecl();
    return isZeroInitializable(rd);
  }

  if (t->getAs<MemberPointerType>()) {
    cgm.errorNYI(SourceLocation(), "isZeroInitializable for MemberPointerType",
                 t);
    return false;
  }

  return true;
}

bool CIRGenTypes::isZeroInitializable(const RecordDecl *rd) {
  return getCIRGenRecordLayout(rd).isZeroInitializable();
}

const CIRGenFunctionInfo &
CIRGenTypes::arrangeCIRFunctionInfo(CanQualType returnType,
                                    llvm::ArrayRef<CanQualType> argTypes,
                                    RequiredArgs required) {
  assert(llvm::all_of(argTypes,
                      [](CanQualType t) { return t.isCanonicalAsParam(); }));
  // Lookup or create unique function info.
  llvm::FoldingSetNodeID id;
  CIRGenFunctionInfo::Profile(id, required, returnType, argTypes);

  void *insertPos = nullptr;
  CIRGenFunctionInfo *fi = functionInfos.FindNodeOrInsertPos(id, insertPos);
  if (fi)
    return *fi;

  assert(!cir::MissingFeatures::opCallCallConv());

  // Construction the function info. We co-allocate the ArgInfos.
  fi = CIRGenFunctionInfo::create(returnType, argTypes, required);
  functionInfos.InsertNode(fi, insertPos);

  return *fi;
}

const CIRGenFunctionInfo &CIRGenTypes::arrangeGlobalDeclaration(GlobalDecl gd) {
  assert(!dyn_cast<ObjCMethodDecl>(gd.getDecl()) &&
         "This is reported as a FIXME in LLVM codegen");
  const auto *fd = cast<FunctionDecl>(gd.getDecl());

  if (isa<CXXConstructorDecl>(gd.getDecl()) ||
      isa<CXXDestructorDecl>(gd.getDecl())) {
    cgm.errorNYI(SourceLocation(),
                 "arrangeGlobalDeclaration for C++ constructor or destructor");
  }

  return arrangeFunctionDeclaration(fd);
}
