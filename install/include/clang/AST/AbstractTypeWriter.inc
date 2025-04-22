/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* A CRTP writer for Clang Type nodes                                         *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TypeProperties.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

template <class PropertyWriter>
class AbstractTypeWriter {
public:
  PropertyWriter &W;

  AbstractTypeWriter(PropertyWriter &W) : W(W) {}

  void write(const Type *node) {
    switch (node->getTypeClass()) {
    case Type::Adjusted:
      return writeAdjustedType(static_cast<const AdjustedType *>(node));
    case Type::Decayed:
      return writeDecayedType(static_cast<const DecayedType *>(node));
    case Type::ConstantArray:
      return writeConstantArrayType(static_cast<const ConstantArrayType *>(node));
    case Type::ArrayParameter:
      return writeArrayParameterType(static_cast<const ArrayParameterType *>(node));
    case Type::DependentSizedArray:
      return writeDependentSizedArrayType(static_cast<const DependentSizedArrayType *>(node));
    case Type::IncompleteArray:
      return writeIncompleteArrayType(static_cast<const IncompleteArrayType *>(node));
    case Type::VariableArray:
      return writeVariableArrayType(static_cast<const VariableArrayType *>(node));
    case Type::Atomic:
      return writeAtomicType(static_cast<const AtomicType *>(node));
    case Type::Attributed:
      return writeAttributedType(static_cast<const AttributedType *>(node));
    case Type::BTFTagAttributed:
      return writeBTFTagAttributedType(static_cast<const BTFTagAttributedType *>(node));
    case Type::BitInt:
      return writeBitIntType(static_cast<const BitIntType *>(node));
    case Type::BlockPointer:
      return writeBlockPointerType(static_cast<const BlockPointerType *>(node));
    case Type::CountAttributed:
      return writeCountAttributedType(static_cast<const CountAttributedType *>(node));
    case Type::Builtin:
      return writeBuiltinType(static_cast<const BuiltinType *>(node));
    case Type::Complex:
      return writeComplexType(static_cast<const ComplexType *>(node));
    case Type::Decltype:
      return writeDecltypeType(static_cast<const DecltypeType *>(node));
    case Type::Auto:
      return writeAutoType(static_cast<const AutoType *>(node));
    case Type::DeducedTemplateSpecialization:
      return writeDeducedTemplateSpecializationType(static_cast<const DeducedTemplateSpecializationType *>(node));
    case Type::DependentAddressSpace:
      return writeDependentAddressSpaceType(static_cast<const DependentAddressSpaceType *>(node));
    case Type::DependentBitInt:
      return writeDependentBitIntType(static_cast<const DependentBitIntType *>(node));
    case Type::DependentName:
      return writeDependentNameType(static_cast<const DependentNameType *>(node));
    case Type::DependentSizedExtVector:
      return writeDependentSizedExtVectorType(static_cast<const DependentSizedExtVectorType *>(node));
    case Type::DependentTemplateSpecialization:
      return writeDependentTemplateSpecializationType(static_cast<const DependentTemplateSpecializationType *>(node));
    case Type::DependentVector:
      return writeDependentVectorType(static_cast<const DependentVectorType *>(node));
    case Type::Elaborated:
      return writeElaboratedType(static_cast<const ElaboratedType *>(node));
    case Type::FunctionNoProto:
      return writeFunctionNoProtoType(static_cast<const FunctionNoProtoType *>(node));
    case Type::FunctionProto:
      return writeFunctionProtoType(static_cast<const FunctionProtoType *>(node));
    case Type::HLSLAttributedResource:
      return writeHLSLAttributedResourceType(static_cast<const HLSLAttributedResourceType *>(node));
    case Type::InjectedClassName:
      return writeInjectedClassNameType(static_cast<const InjectedClassNameType *>(node));
    case Type::MacroQualified:
      return writeMacroQualifiedType(static_cast<const MacroQualifiedType *>(node));
    case Type::ConstantMatrix:
      return writeConstantMatrixType(static_cast<const ConstantMatrixType *>(node));
    case Type::DependentSizedMatrix:
      return writeDependentSizedMatrixType(static_cast<const DependentSizedMatrixType *>(node));
    case Type::MemberPointer:
      return writeMemberPointerType(static_cast<const MemberPointerType *>(node));
    case Type::ObjCObjectPointer:
      return writeObjCObjectPointerType(static_cast<const ObjCObjectPointerType *>(node));
    case Type::ObjCObject:
      return writeObjCObjectType(static_cast<const ObjCObjectType *>(node));
    case Type::ObjCInterface:
      return writeObjCInterfaceType(static_cast<const ObjCInterfaceType *>(node));
    case Type::ObjCTypeParam:
      return writeObjCTypeParamType(static_cast<const ObjCTypeParamType *>(node));
    case Type::PackExpansion:
      return writePackExpansionType(static_cast<const PackExpansionType *>(node));
    case Type::PackIndexing:
      return writePackIndexingType(static_cast<const PackIndexingType *>(node));
    case Type::Paren:
      return writeParenType(static_cast<const ParenType *>(node));
    case Type::Pipe:
      return writePipeType(static_cast<const PipeType *>(node));
    case Type::Pointer:
      return writePointerType(static_cast<const PointerType *>(node));
    case Type::LValueReference:
      return writeLValueReferenceType(static_cast<const LValueReferenceType *>(node));
    case Type::RValueReference:
      return writeRValueReferenceType(static_cast<const RValueReferenceType *>(node));
    case Type::SubstTemplateTypeParmPack:
      return writeSubstTemplateTypeParmPackType(static_cast<const SubstTemplateTypeParmPackType *>(node));
    case Type::SubstTemplateTypeParm:
      return writeSubstTemplateTypeParmType(static_cast<const SubstTemplateTypeParmType *>(node));
    case Type::Enum:
      return writeEnumType(static_cast<const EnumType *>(node));
    case Type::Record:
      return writeRecordType(static_cast<const RecordType *>(node));
    case Type::TemplateSpecialization:
      return writeTemplateSpecializationType(static_cast<const TemplateSpecializationType *>(node));
    case Type::TemplateTypeParm:
      return writeTemplateTypeParmType(static_cast<const TemplateTypeParmType *>(node));
    case Type::TypeOfExpr:
      return writeTypeOfExprType(static_cast<const TypeOfExprType *>(node));
    case Type::TypeOf:
      return writeTypeOfType(static_cast<const TypeOfType *>(node));
    case Type::Typedef:
      return writeTypedefType(static_cast<const TypedefType *>(node));
    case Type::UnaryTransform:
      return writeUnaryTransformType(static_cast<const UnaryTransformType *>(node));
    case Type::UnresolvedUsing:
      return writeUnresolvedUsingType(static_cast<const UnresolvedUsingType *>(node));
    case Type::Using:
      return writeUsingType(static_cast<const UsingType *>(node));
    case Type::Vector:
      return writeVectorType(static_cast<const VectorType *>(node));
    case Type::ExtVector:
      return writeExtVectorType(static_cast<const ExtVectorType *>(node));
    }
    llvm_unreachable("bad kind");
  }

  void writeAdjustedType(const AdjustedType *node) {
    QualType originalType = ( node->getOriginalType() );
    W.find("originalType").writeQualType(originalType);
    QualType adjustedType = ( node->getAdjustedType() );
    W.find("adjustedType").writeQualType(adjustedType);
  }

  void writeDecayedType(const DecayedType *node) {
    QualType originalType = ( node->getOriginalType() );
    W.find("originalType").writeQualType(originalType);
  }

  void writeConstantArrayType(const ConstantArrayType *node) {
    llvm::APInt sizeValue = ( node->getSize() );
    W.find("sizeValue").writeAPInt(sizeValue);
    const Expr* size = ( node->getSizeExpr() );
    W.find("size").writeExprRef(size);
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    ArraySizeModifier sizeModifier = ( node->getSizeModifier() );
    W.find("sizeModifier").writeArraySizeModifier(sizeModifier);
    Qualifiers indexQualifiers = ( Qualifiers::fromCVRMask(node->getIndexTypeCVRQualifiers()) );
    W.find("indexQualifiers").writeQualifiers(indexQualifiers);
  }

  void writeArrayParameterType(const ArrayParameterType *node) {
    llvm::APInt sizeValue = ( node->getSize() );
    W.find("sizeValue").writeAPInt(sizeValue);
    const Expr* size = ( node->getSizeExpr() );
    W.find("size").writeExprRef(size);
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    ArraySizeModifier sizeModifier = ( node->getSizeModifier() );
    W.find("sizeModifier").writeArraySizeModifier(sizeModifier);
    Qualifiers indexQualifiers = ( Qualifiers::fromCVRMask(node->getIndexTypeCVRQualifiers()) );
    W.find("indexQualifiers").writeQualifiers(indexQualifiers);
  }

  void writeDependentSizedArrayType(const DependentSizedArrayType *node) {
    const Expr* size = ( node->getSizeExpr() );
    W.find("size").writeExprRef(size);
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    ArraySizeModifier sizeModifier = ( node->getSizeModifier() );
    W.find("sizeModifier").writeArraySizeModifier(sizeModifier);
    Qualifiers indexQualifiers = ( Qualifiers::fromCVRMask(node->getIndexTypeCVRQualifiers()) );
    W.find("indexQualifiers").writeQualifiers(indexQualifiers);
  }

  void writeIncompleteArrayType(const IncompleteArrayType *node) {
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    ArraySizeModifier sizeModifier = ( node->getSizeModifier() );
    W.find("sizeModifier").writeArraySizeModifier(sizeModifier);
    Qualifiers indexQualifiers = ( Qualifiers::fromCVRMask(node->getIndexTypeCVRQualifiers()) );
    W.find("indexQualifiers").writeQualifiers(indexQualifiers);
  }

  void writeVariableArrayType(const VariableArrayType *node) {
    const Expr* size = ( node->getSizeExpr() );
    W.find("size").writeExprRef(size);
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    ArraySizeModifier sizeModifier = ( node->getSizeModifier() );
    W.find("sizeModifier").writeArraySizeModifier(sizeModifier);
    Qualifiers indexQualifiers = ( Qualifiers::fromCVRMask(node->getIndexTypeCVRQualifiers()) );
    W.find("indexQualifiers").writeQualifiers(indexQualifiers);
  }

  void writeAtomicType(const AtomicType *node) {
    QualType valueType = ( node->getValueType() );
    W.find("valueType").writeQualType(valueType);
  }

  void writeAttributedType(const AttributedType *node) {
    QualType modifiedType = ( node->getModifiedType() );
    W.find("modifiedType").writeQualType(modifiedType);
    QualType equivalentType = ( node->getEquivalentType() );
    W.find("equivalentType").writeQualType(equivalentType);
    attr::Kind attrKind = ( node->getAttrKind() );
    W.find("attrKind").writeAttrKind(attrKind);
    const Attr * attribute = ( node->getAttr() );
    W.find("attribute").writeAttr(attribute);
  }

  void writeBTFTagAttributedType(const BTFTagAttributedType *node) {
    const BTFTypeTagAttr * attr = ( node->getAttr() );
    W.find("attr").writeBTFTypeTagAttr(attr);
    QualType wrappedType = ( node->getWrappedType() );
    W.find("wrappedType").writeQualType(wrappedType);
  }

  void writeBitIntType(const BitIntType *node) {
    bool isUnsigned = ( node->isUnsigned() );
    W.find("isUnsigned").writeBool(isUnsigned);
    uint32_t numBits = ( node->getNumBits() );
    W.find("numBits").writeUInt32(numBits);
  }

  void writeBlockPointerType(const BlockPointerType *node) {
    QualType pointeeType = ( node->getPointeeType() );
    W.find("pointeeType").writeQualType(pointeeType);
  }

  void writeCountAttributedType(const CountAttributedType *node) {
    QualType WrappedTy = ( node->desugar() );
    W.find("WrappedTy").writeQualType(WrappedTy);
    const Expr* CountExpr = ( node->getCountExpr() );
    W.find("CountExpr").writeExprRef(CountExpr);
    bool CountInBytes = ( node->isCountInBytes() );
    W.find("CountInBytes").writeBool(CountInBytes);
    bool OrNull = ( node->isOrNull() );
    W.find("OrNull").writeBool(OrNull);
    llvm::ArrayRef<TypeCoupledDeclRefInfo> CoupledDecls = ( node->getCoupledDecls() );
    W.find("CoupledDecls").writeArray(CoupledDecls);
  }

  void writeBuiltinType(const BuiltinType *node) {
    BuiltinType::Kind kind = ( node->getKind() );
    W.find("kind").writeBuiltinTypeKind(kind);
  }

  void writeComplexType(const ComplexType *node) {
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
  }

  void writeDecltypeType(const DecltypeType *node) {
    QualType underlyingType = ( node->getUnderlyingType() );
    W.find("underlyingType").writeQualType(underlyingType);
    const Expr* expression = ( node->getUnderlyingExpr() );
    W.find("expression").writeExprRef(expression);
  }

  void writeAutoType(const AutoType *node) {
    std::optional<QualType> deducedType = ( makeOptionalFromNullable(node->getDeducedType()) );
    W.find("deducedType").writeOptional(deducedType);
    AutoTypeKeyword keyword = ( node->getKeyword() );
    W.find("keyword").writeAutoTypeKeyword(keyword);
    std::optional<const ConceptDecl*> typeConstraintConcept = ( makeOptionalFromPointer(
        const_cast<const ConceptDecl*>(node->getTypeConstraintConcept())) );
    W.find("typeConstraintConcept").writeOptional(typeConstraintConcept);
    llvm::ArrayRef<TemplateArgument> typeConstraintArguments = ( node->getTypeConstraintArguments() );
    W.find("typeConstraintArguments").writeArray(typeConstraintArguments);
    uint32_t dependence = ( !node->getDeducedType().isNull() ? 0 :
                  node->containsUnexpandedParameterPack() ? 2 :
                  node->isDependentType() ? 1 : 0 );
    W.find("dependence").writeUInt32(dependence);
  }

  void writeDeducedTemplateSpecializationType(const DeducedTemplateSpecializationType *node) {
    std::optional<TemplateName> templateName = ( makeOptionalFromNullable(node->getTemplateName()) );
    W.find("templateName").writeOptional(templateName);
    QualType deducedType = ( node->getDeducedType() );
    W.find("deducedType").writeQualType(deducedType);
    bool dependent = ( !node->getDeducedType().isNull()
                    ? false : node->isDependentType() );
    W.find("dependent").writeBool(dependent);
  }

  void writeDependentAddressSpaceType(const DependentAddressSpaceType *node) {
    QualType pointeeType = ( node->getPointeeType() );
    W.find("pointeeType").writeQualType(pointeeType);
    const Expr* addressSpace = ( node->getAddrSpaceExpr() );
    W.find("addressSpace").writeExprRef(addressSpace);
    SourceLocation attributeLoc = ( node->getAttributeLoc() );
    W.find("attributeLoc").writeSourceLocation(attributeLoc);
  }

  void writeDependentBitIntType(const DependentBitIntType *node) {
    bool isUnsigned = ( node->isUnsigned() );
    W.find("isUnsigned").writeBool(isUnsigned);
    const Expr* numBitsExpr = ( node->getNumBitsExpr() );
    W.find("numBitsExpr").writeExprRef(numBitsExpr);
  }

  void writeDependentNameType(const DependentNameType *node) {
    ElaboratedTypeKeyword keyword = ( node->getKeyword() );
    W.find("keyword").writeElaboratedTypeKeyword(keyword);
    NestedNameSpecifier * qualifier = ( node->getQualifier() );
    W.find("qualifier").writeNestedNameSpecifier(qualifier);
    const IdentifierInfo* name = ( node->getIdentifier() );
    W.find("name").writeIdentifier(name);
  }

  void writeDependentSizedExtVectorType(const DependentSizedExtVectorType *node) {
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    const Expr* size = ( node->getSizeExpr() );
    W.find("size").writeExprRef(size);
    SourceLocation attributeLoc = ( node->getAttributeLoc() );
    W.find("attributeLoc").writeSourceLocation(attributeLoc);
  }

  void writeDependentTemplateSpecializationType(const DependentTemplateSpecializationType *node) {
    
    const auto &dtn = node->getDependentTemplateName();
    auto name = dtn.getName();
  
    NestedNameSpecifier * qualifier = ( dtn.getQualifier() );
    W.find("qualifier").writeNestedNameSpecifier(qualifier);
    std::optional<const IdentifierInfo*> identifier = ( makeOptionalFromPointer(name.getIdentifier()) );
    W.find("identifier").writeOptional(identifier);
    if ( !identifier ) {
    OverloadedOperatorKind operatorKind = ( name.getOperator() );
    W.find("operatorKind").writeOverloadedOperatorKind(operatorKind);
    }
    bool HasTemplateKeyword = ( dtn.hasTemplateKeyword() );
    W.find("HasTemplateKeyword").writeBool(HasTemplateKeyword);
    ElaboratedTypeKeyword keyword = ( node->getKeyword() );
    W.find("keyword").writeElaboratedTypeKeyword(keyword);
    llvm::ArrayRef<TemplateArgument> templateArguments = ( node->template_arguments() );
    W.find("templateArguments").writeArray(templateArguments);
  }

  void writeDependentVectorType(const DependentVectorType *node) {
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    const Expr* size = ( node->getSizeExpr() );
    W.find("size").writeExprRef(size);
    SourceLocation attributeLoc = ( node->getAttributeLoc() );
    W.find("attributeLoc").writeSourceLocation(attributeLoc);
    VectorKind vectorKind = ( node->getVectorKind() );
    W.find("vectorKind").writeVectorKind(vectorKind);
  }

  void writeElaboratedType(const ElaboratedType *node) {
    ElaboratedTypeKeyword keyword = ( node->getKeyword() );
    W.find("keyword").writeElaboratedTypeKeyword(keyword);
    NestedNameSpecifier * qualifier = ( node->getQualifier() );
    W.find("qualifier").writeNestedNameSpecifier(qualifier);
    QualType namedType = ( node->getNamedType() );
    W.find("namedType").writeQualType(namedType);
    std::optional<const TagDecl*> ownedTag = ( makeOptionalFromPointer(
                    const_cast<const TagDecl *>(node->getOwnedTagDecl())) );
    W.find("ownedTag").writeOptional(ownedTag);
  }

  void writeFunctionNoProtoType(const FunctionNoProtoType *node) {
    QualType returnType = ( node->getReturnType() );
    W.find("returnType").writeQualType(returnType);
    bool noReturn = ( node->getExtInfo().getNoReturn() );
    W.find("noReturn").writeBool(noReturn);
    bool hasRegParm = ( node->getExtInfo().getHasRegParm() );
    W.find("hasRegParm").writeBool(hasRegParm);
    uint32_t regParm = ( node->getExtInfo().getRegParm() );
    W.find("regParm").writeUInt32(regParm);
    CallingConv callingConvention = ( node->getExtInfo().getCC() );
    W.find("callingConvention").writeCallingConv(callingConvention);
    bool producesResult = ( node->getExtInfo().getProducesResult() );
    W.find("producesResult").writeBool(producesResult);
    bool noCallerSavedRegs = ( node->getExtInfo().getNoCallerSavedRegs() );
    W.find("noCallerSavedRegs").writeBool(noCallerSavedRegs);
    bool noCfCheck = ( node->getExtInfo().getNoCfCheck() );
    W.find("noCfCheck").writeBool(noCfCheck);
    bool cmseNSCall = ( node->getExtInfo().getCmseNSCall() );
    W.find("cmseNSCall").writeBool(cmseNSCall);
  }

  void writeFunctionProtoType(const FunctionProtoType *node) {
    bool variadic = ( node->isVariadic() );
    W.find("variadic").writeBool(variadic);
    bool trailingReturn = ( node->hasTrailingReturn() );
    W.find("trailingReturn").writeBool(trailingReturn);
    Qualifiers methodQualifiers = ( node->getMethodQuals() );
    W.find("methodQualifiers").writeQualifiers(methodQualifiers);
    RefQualifierKind refQualifier = ( node->getRefQualifier() );
    W.find("refQualifier").writeRefQualifierKind(refQualifier);
    FunctionProtoType::ExceptionSpecInfo exceptionSpecifier = ( node->getExceptionSpecInfo() );
    W.find("exceptionSpecifier").writeExceptionSpecInfo(exceptionSpecifier);
    llvm::ArrayRef<QualType> parameters = ( node->getParamTypes() );
    W.find("parameters").writeArray(parameters);
    llvm::ArrayRef<FunctionProtoType::ExtParameterInfo> extParameterInfo = ( node->hasExtParameterInfos()
                    ? node->getExtParameterInfos()
                    : llvm::ArrayRef<FunctionProtoType::ExtParameterInfo>() );
    W.find("extParameterInfo").writeArray(extParameterInfo);
    uint32_t AArch64SMEAttributes = ( node->getAArch64SMEAttributes() );
    W.find("AArch64SMEAttributes").writeUInt32(AArch64SMEAttributes);
    llvm::ArrayRef<FunctionEffect> functionEffects = ( node->getFunctionEffectsWithoutConditions() );
    W.find("functionEffects").writeArray(functionEffects);
    llvm::ArrayRef<EffectConditionExpr> functionEffectConds = ( node->getFunctionEffectConditions() );
    W.find("functionEffectConds").writeArray(functionEffectConds);
    QualType returnType = ( node->getReturnType() );
    W.find("returnType").writeQualType(returnType);
    bool noReturn = ( node->getExtInfo().getNoReturn() );
    W.find("noReturn").writeBool(noReturn);
    bool hasRegParm = ( node->getExtInfo().getHasRegParm() );
    W.find("hasRegParm").writeBool(hasRegParm);
    uint32_t regParm = ( node->getExtInfo().getRegParm() );
    W.find("regParm").writeUInt32(regParm);
    CallingConv callingConvention = ( node->getExtInfo().getCC() );
    W.find("callingConvention").writeCallingConv(callingConvention);
    bool producesResult = ( node->getExtInfo().getProducesResult() );
    W.find("producesResult").writeBool(producesResult);
    bool noCallerSavedRegs = ( node->getExtInfo().getNoCallerSavedRegs() );
    W.find("noCallerSavedRegs").writeBool(noCallerSavedRegs);
    bool noCfCheck = ( node->getExtInfo().getNoCfCheck() );
    W.find("noCfCheck").writeBool(noCfCheck);
    bool cmseNSCall = ( node->getExtInfo().getCmseNSCall() );
    W.find("cmseNSCall").writeBool(cmseNSCall);
  }

  void writeHLSLAttributedResourceType(const HLSLAttributedResourceType *node) {
    uint32_t resClass = ( static_cast<uint32_t>(node->getAttrs().ResourceClass) );
    W.find("resClass").writeUInt32(resClass);
    bool isROV = ( node->getAttrs().IsROV );
    W.find("isROV").writeBool(isROV);
    bool rawBuffer = ( node->getAttrs().RawBuffer );
    W.find("rawBuffer").writeBool(rawBuffer);
    QualType wrappedTy = ( node->getWrappedType() );
    W.find("wrappedTy").writeQualType(wrappedTy);
    QualType containedTy = ( node->getContainedType() );
    W.find("containedTy").writeQualType(containedTy);
  }

  void writeInjectedClassNameType(const InjectedClassNameType *node) {
    const Decl* declaration = ( node->getDecl()->getCanonicalDecl() );
    W.find("declaration").writeDeclRef(declaration);
    QualType injectedSpecializationType = ( node->getInjectedSpecializationType() );
    W.find("injectedSpecializationType").writeQualType(injectedSpecializationType);
  }

  void writeMacroQualifiedType(const MacroQualifiedType *node) {
    QualType underlyingType = ( node->getUnderlyingType() );
    W.find("underlyingType").writeQualType(underlyingType);
    const IdentifierInfo* macroIdentifier = ( node->getMacroIdentifier() );
    W.find("macroIdentifier").writeIdentifier(macroIdentifier);
  }

  void writeConstantMatrixType(const ConstantMatrixType *node) {
    uint32_t numRows = ( node->getNumRows() );
    W.find("numRows").writeUInt32(numRows);
    uint32_t numColumns = ( node->getNumColumns() );
    W.find("numColumns").writeUInt32(numColumns);
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
  }

  void writeDependentSizedMatrixType(const DependentSizedMatrixType *node) {
    const Expr* rows = ( node->getRowExpr() );
    W.find("rows").writeExprRef(rows);
    const Expr* columns = ( node->getColumnExpr() );
    W.find("columns").writeExprRef(columns);
    SourceLocation attributeLoc = ( node->getAttributeLoc() );
    W.find("attributeLoc").writeSourceLocation(attributeLoc);
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
  }

  void writeMemberPointerType(const MemberPointerType *node) {
    QualType pointeeType = ( node->getPointeeType() );
    W.find("pointeeType").writeQualType(pointeeType);
    NestedNameSpecifier * Qualifier = ( node->getQualifier() );
    W.find("Qualifier").writeNestedNameSpecifier(Qualifier);
    const Decl* Cls = ( node->getMostRecentCXXRecordDecl() );
    W.find("Cls").writeDeclRef(Cls);
  }

  void writeObjCObjectPointerType(const ObjCObjectPointerType *node) {
    QualType pointeeType = ( node->getPointeeType() );
    W.find("pointeeType").writeQualType(pointeeType);
  }

  void writeObjCObjectType(const ObjCObjectType *node) {
    QualType baseType = ( node->getBaseType() );
    W.find("baseType").writeQualType(baseType);
    llvm::ArrayRef<QualType> typeArgsAsWritten = ( node->getTypeArgsAsWritten() );
    W.find("typeArgsAsWritten").writeArray(typeArgsAsWritten);
    llvm::ArrayRef<const ObjCProtocolDecl*> qualifiers = ( node->getProtocols() );
    W.find("qualifiers").writeArray(qualifiers);
    bool isKindOfTypeAsWritten = ( node->isKindOfTypeAsWritten() );
    W.find("isKindOfTypeAsWritten").writeBool(isKindOfTypeAsWritten);
  }

  void writeObjCInterfaceType(const ObjCInterfaceType *node) {
    const Decl* declaration = ( node->getDecl()->getCanonicalDecl() );
    W.find("declaration").writeDeclRef(declaration);
  }

  void writeObjCTypeParamType(const ObjCTypeParamType *node) {
    const ObjCTypeParamDecl* declaration = ( node->getDecl() );
    W.find("declaration").writeObjCTypeParamDeclRef(declaration);
    llvm::ArrayRef<const ObjCProtocolDecl*> qualifiers = ( node->getProtocols() );
    W.find("qualifiers").writeArray(qualifiers);
  }

  void writePackExpansionType(const PackExpansionType *node) {
    QualType pattern = ( node->getPattern() );
    W.find("pattern").writeQualType(pattern);
    UnsignedOrNone numExpansions = ( node->getNumExpansions() );
    W.find("numExpansions").writeUnsignedOrNone(numExpansions);
  }

  void writePackIndexingType(const PackIndexingType *node) {
    QualType pattern = ( node->getPattern() );
    W.find("pattern").writeQualType(pattern);
    const Expr* indexExpression = ( node->getIndexExpr() );
    W.find("indexExpression").writeExprRef(indexExpression);
    bool isFullySubstituted = ( node->isFullySubstituted() );
    W.find("isFullySubstituted").writeBool(isFullySubstituted);
  }

  void writeParenType(const ParenType *node) {
    QualType innerType = ( node->getInnerType() );
    W.find("innerType").writeQualType(innerType);
  }

  void writePipeType(const PipeType *node) {
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    bool isReadOnly = ( node->isReadOnly() );
    W.find("isReadOnly").writeBool(isReadOnly);
  }

  void writePointerType(const PointerType *node) {
    QualType pointeeType = ( node->getPointeeType() );
    W.find("pointeeType").writeQualType(pointeeType);
  }

  void writeLValueReferenceType(const LValueReferenceType *node) {
    bool isSpelledAsLValue = ( node->isSpelledAsLValue() );
    W.find("isSpelledAsLValue").writeBool(isSpelledAsLValue);
    QualType pointeeTypeAsWritten = ( node->getPointeeTypeAsWritten() );
    W.find("pointeeTypeAsWritten").writeQualType(pointeeTypeAsWritten);
  }

  void writeRValueReferenceType(const RValueReferenceType *node) {
    QualType pointeeTypeAsWritten = ( node->getPointeeTypeAsWritten() );
    W.find("pointeeTypeAsWritten").writeQualType(pointeeTypeAsWritten);
  }

  void writeSubstTemplateTypeParmPackType(const SubstTemplateTypeParmPackType *node) {
    const Decl* associatedDecl = ( node->getAssociatedDecl() );
    W.find("associatedDecl").writeDeclRef(associatedDecl);
    uint32_t Index = ( node->getIndex() );
    W.find("Index").writeUInt32(Index);
    bool Final = ( node->getFinal() );
    W.find("Final").writeBool(Final);
    TemplateArgument replacementPack = ( node->getArgumentPack() );
    W.find("replacementPack").writeTemplateArgument(replacementPack);
  }

  void writeSubstTemplateTypeParmType(const SubstTemplateTypeParmType *node) {
    QualType replacementType = ( node->getReplacementType() );
    W.find("replacementType").writeQualType(replacementType);
    const Decl* associatedDecl = ( node->getAssociatedDecl() );
    W.find("associatedDecl").writeDeclRef(associatedDecl);
    uint32_t Index = ( node->getIndex() );
    W.find("Index").writeUInt32(Index);
    UnsignedOrNone PackIndex = ( node->getPackIndex() );
    W.find("PackIndex").writeUnsignedOrNone(PackIndex);
    bool Final = ( node->getFinal() );
    W.find("Final").writeBool(Final);
  }

  void writeEnumType(const EnumType *node) {
    bool dependent = ( node->isDependentType() );
    W.find("dependent").writeBool(dependent);
    const Decl* declaration = ( node->getDecl()->getCanonicalDecl() );
    W.find("declaration").writeDeclRef(declaration);
  }

  void writeRecordType(const RecordType *node) {
    bool dependent = ( node->isDependentType() );
    W.find("dependent").writeBool(dependent);
    const Decl* declaration = ( node->getDecl()->getCanonicalDecl() );
    W.find("declaration").writeDeclRef(declaration);
  }

  void writeTemplateSpecializationType(const TemplateSpecializationType *node) {
    TemplateName templateName = ( node->getTemplateName() );
    W.find("templateName").writeTemplateName(templateName);
    llvm::ArrayRef<TemplateArgument> args = ( node->template_arguments() );
    W.find("args").writeArray(args);
    QualType UnderlyingType = ( node->isCanonicalUnqualified() ? QualType() :
                                                   node->desugar() );
    W.find("UnderlyingType").writeQualType(UnderlyingType);
  }

  void writeTemplateTypeParmType(const TemplateTypeParmType *node) {
    uint32_t depth = ( node->getDepth() );
    W.find("depth").writeUInt32(depth);
    uint32_t index = ( node->getIndex() );
    W.find("index").writeUInt32(index);
    bool isParameterPack = ( node->isParameterPack() );
    W.find("isParameterPack").writeBool(isParameterPack);
    std::optional<const TemplateTypeParmDecl*> declaration = ( makeOptionalFromPointer(
                    const_cast<const TemplateTypeParmDecl*>(node->getDecl())) );
    W.find("declaration").writeOptional(declaration);
  }

  void writeTypeOfExprType(const TypeOfExprType *node) {
    const Expr* expression = ( node->getUnderlyingExpr() );
    W.find("expression").writeExprRef(expression);
    TypeOfKind kind = ( node->getKind() );
    W.find("kind").writeTypeOfKind(kind);
  }

  void writeTypeOfType(const TypeOfType *node) {
    QualType unmodifiedType = ( node->getUnmodifiedType() );
    W.find("unmodifiedType").writeQualType(unmodifiedType);
    TypeOfKind kind = ( node->getKind() );
    W.find("kind").writeTypeOfKind(kind);
  }

  void writeTypedefType(const TypedefType *node) {
    const Decl* declaration = ( node->getDecl() );
    W.find("declaration").writeDeclRef(declaration);
    QualType underlyingType = ( node->desugar() );
    W.find("underlyingType").writeQualType(underlyingType);
  }

  void writeUnaryTransformType(const UnaryTransformType *node) {
    QualType baseType = ( node->getBaseType() );
    W.find("baseType").writeQualType(baseType);
    QualType underlyingType = ( node->getUnderlyingType() );
    W.find("underlyingType").writeQualType(underlyingType);
    UnaryTransformType::UTTKind transform = ( node->getUTTKind() );
    W.find("transform").writeUnaryTypeTransformKind(transform);
  }

  void writeUnresolvedUsingType(const UnresolvedUsingType *node) {
    const Decl* declaration = ( node->getDecl() );
    W.find("declaration").writeDeclRef(declaration);
  }

  void writeUsingType(const UsingType *node) {
    const UsingShadowDecl* foundDeclaration = ( node->getFoundDecl() );
    W.find("foundDeclaration").writeUsingShadowDeclRef(foundDeclaration);
    QualType underlyingType = ( node->getUnderlyingType() );
    W.find("underlyingType").writeQualType(underlyingType);
  }

  void writeVectorType(const VectorType *node) {
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    uint32_t numElements = ( node->getNumElements() );
    W.find("numElements").writeUInt32(numElements);
    VectorKind vectorKind = ( node->getVectorKind() );
    W.find("vectorKind").writeVectorKind(vectorKind);
  }

  void writeExtVectorType(const ExtVectorType *node) {
    QualType elementType = ( node->getElementType() );
    W.find("elementType").writeQualType(elementType);
    uint32_t numElements = ( node->getNumElements() );
    W.find("numElements").writeUInt32(numElements);
  }

};

