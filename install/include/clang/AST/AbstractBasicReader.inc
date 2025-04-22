/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Helper classes for BasicReaders                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: PropertiesBase.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

template <class ValueType>
struct ReadDispatcher;
template <>
struct ReadDispatcher<llvm::APInt> {
  template <class BasicReader, class... Args>
  static llvm::APInt read(BasicReader &R, Args &&... args) {
    return R.readAPInt(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<llvm::APSInt> {
  template <class BasicReader, class... Args>
  static llvm::APSInt read(BasicReader &R, Args &&... args) {
    return R.readAPSInt(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<APValue> {
  template <class BasicReader, class... Args>
  static APValue read(BasicReader &R, Args &&... args) {
    return R.readAPValue(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<APValue::ValueKind> {
  template <class BasicReader, class... Args>
  static APValue::ValueKind read(BasicReader &R, Args &&... args) {
    return R.readAPValueKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<ArraySizeModifier> {
  template <class BasicReader, class... Args>
  static ArraySizeModifier read(BasicReader &R, Args &&... args) {
    return R.readArraySizeModifier(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<const Attr *> {
  template <class BasicReader, class... Args>
  static const Attr * read(BasicReader &R, Args &&... args) {
    return R.readAttr(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<attr::Kind> {
  template <class BasicReader, class... Args>
  static attr::Kind read(BasicReader &R, Args &&... args) {
    return R.readAttrKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<AutoTypeKeyword> {
  template <class BasicReader, class... Args>
  static AutoTypeKeyword read(BasicReader &R, Args &&... args) {
    return R.readAutoTypeKeyword(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<const BTFTypeTagAttr *> {
  template <class BasicReader, class... Args>
  static const BTFTypeTagAttr * read(BasicReader &R, Args &&... args) {
    return R.readBTFTypeTagAttr(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<bool> {
  template <class BasicReader, class... Args>
  static bool read(BasicReader &R, Args &&... args) {
    return R.readBool(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<BuiltinType::Kind> {
  template <class BasicReader, class... Args>
  static BuiltinType::Kind read(BasicReader &R, Args &&... args) {
    return R.readBuiltinTypeKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<CXXRecordDecl*> {
  template <class BasicReader, class... Args>
  static CXXRecordDecl* read(BasicReader &R, Args &&... args) {
    return R.readCXXRecordDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<CallingConv> {
  template <class BasicReader, class... Args>
  static CallingConv read(BasicReader &R, Args &&... args) {
    return R.readCallingConv(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<ConceptDecl*> {
  template <class BasicReader, class... Args>
  static ConceptDecl* read(BasicReader &R, Args &&... args) {
    return R.readConceptDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<Decl*> {
  template <class BasicReader, class... Args>
  static Decl* read(BasicReader &R, Args &&... args) {
    return R.readDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<DeclarationName> {
  template <class BasicReader, class... Args>
  static DeclarationName read(BasicReader &R, Args &&... args) {
    return R.readDeclarationName(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<DeclarationName::NameKind> {
  template <class BasicReader, class... Args>
  static DeclarationName::NameKind read(BasicReader &R, Args &&... args) {
    return R.readDeclarationNameKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<EffectConditionExpr> {
  template <class BasicReader, class... Args>
  static EffectConditionExpr read(BasicReader &R, Args &&... args) {
    return R.readEffectConditionExpr(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<ElaboratedTypeKeyword> {
  template <class BasicReader, class... Args>
  static ElaboratedTypeKeyword read(BasicReader &R, Args &&... args) {
    return R.readElaboratedTypeKeyword(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<FunctionProtoType::ExceptionSpecInfo> {
  template <class BasicReader, class... Args>
  static FunctionProtoType::ExceptionSpecInfo read(BasicReader &R, Args &&... args) {
    return R.readExceptionSpecInfo(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<Expr*> {
  template <class BasicReader, class... Args>
  static Expr* read(BasicReader &R, Args &&... args) {
    return R.readExprRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<FunctionProtoType::ExtParameterInfo> {
  template <class BasicReader, class... Args>
  static FunctionProtoType::ExtParameterInfo read(BasicReader &R, Args &&... args) {
    return R.readExtParameterInfo(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<llvm::FixedPointSemantics> {
  template <class BasicReader, class... Args>
  static llvm::FixedPointSemantics read(BasicReader &R, Args &&... args) {
    return R.readFixedPointSemantics(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<FunctionDecl*> {
  template <class BasicReader, class... Args>
  static FunctionDecl* read(BasicReader &R, Args &&... args) {
    return R.readFunctionDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<FunctionEffect> {
  template <class BasicReader, class... Args>
  static FunctionEffect read(BasicReader &R, Args &&... args) {
    return R.readFunctionEffect(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<IdentifierInfo*> {
  template <class BasicReader, class... Args>
  static IdentifierInfo* read(BasicReader &R, Args &&... args) {
    return R.readIdentifier(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<APValue::LValuePathEntry> {
  template <class BasicReader, class... Args>
  static APValue::LValuePathEntry read(BasicReader &R, Args &&... args) {
    return R.readLValuePathEntry(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<APValue::LValuePathSerializationHelper> {
  template <class BasicReader, class... Args>
  static APValue::LValuePathSerializationHelper read(BasicReader &R, Args &&... args) {
    return R.readLValuePathSerializationHelper(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<NamedDecl*> {
  template <class BasicReader, class... Args>
  static NamedDecl* read(BasicReader &R, Args &&... args) {
    return R.readNamedDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<NamespaceAliasDecl*> {
  template <class BasicReader, class... Args>
  static NamespaceAliasDecl* read(BasicReader &R, Args &&... args) {
    return R.readNamespaceAliasDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<NamespaceDecl*> {
  template <class BasicReader, class... Args>
  static NamespaceDecl* read(BasicReader &R, Args &&... args) {
    return R.readNamespaceDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<NestedNameSpecifier *> {
  template <class BasicReader, class... Args>
  static NestedNameSpecifier * read(BasicReader &R, Args &&... args) {
    return R.readNestedNameSpecifier(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<NestedNameSpecifier::SpecifierKind> {
  template <class BasicReader, class... Args>
  static NestedNameSpecifier::SpecifierKind read(BasicReader &R, Args &&... args) {
    return R.readNestedNameSpecifierKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<ObjCProtocolDecl*> {
  template <class BasicReader, class... Args>
  static ObjCProtocolDecl* read(BasicReader &R, Args &&... args) {
    return R.readObjCProtocolDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<ObjCTypeParamDecl*> {
  template <class BasicReader, class... Args>
  static ObjCTypeParamDecl* read(BasicReader &R, Args &&... args) {
    return R.readObjCTypeParamDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<OverloadedOperatorKind> {
  template <class BasicReader, class... Args>
  static OverloadedOperatorKind read(BasicReader &R, Args &&... args) {
    return R.readOverloadedOperatorKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<QualType> {
  template <class BasicReader, class... Args>
  static QualType read(BasicReader &R, Args &&... args) {
    return R.readQualType(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<Qualifiers> {
  template <class BasicReader, class... Args>
  static Qualifiers read(BasicReader &R, Args &&... args) {
    return R.readQualifiers(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<RefQualifierKind> {
  template <class BasicReader, class... Args>
  static RefQualifierKind read(BasicReader &R, Args &&... args) {
    return R.readRefQualifierKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<Selector> {
  template <class BasicReader, class... Args>
  static Selector read(BasicReader &R, Args &&... args) {
    return R.readSelector(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<SourceLocation> {
  template <class BasicReader, class... Args>
  static SourceLocation read(BasicReader &R, Args &&... args) {
    return R.readSourceLocation(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<Stmt*> {
  template <class BasicReader, class... Args>
  static Stmt* read(BasicReader &R, Args &&... args) {
    return R.readStmtRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TagDecl*> {
  template <class BasicReader, class... Args>
  static TagDecl* read(BasicReader &R, Args &&... args) {
    return R.readTagDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TemplateArgument> {
  template <class BasicReader, class... Args>
  static TemplateArgument read(BasicReader &R, Args &&... args) {
    return R.readTemplateArgument(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TemplateArgument::ArgKind> {
  template <class BasicReader, class... Args>
  static TemplateArgument::ArgKind read(BasicReader &R, Args &&... args) {
    return R.readTemplateArgumentKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TemplateDecl*> {
  template <class BasicReader, class... Args>
  static TemplateDecl* read(BasicReader &R, Args &&... args) {
    return R.readTemplateDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TemplateName> {
  template <class BasicReader, class... Args>
  static TemplateName read(BasicReader &R, Args &&... args) {
    return R.readTemplateName(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TemplateName::NameKind> {
  template <class BasicReader, class... Args>
  static TemplateName::NameKind read(BasicReader &R, Args &&... args) {
    return R.readTemplateNameKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TemplateTemplateParmDecl*> {
  template <class BasicReader, class... Args>
  static TemplateTemplateParmDecl* read(BasicReader &R, Args &&... args) {
    return R.readTemplateTemplateParmDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TemplateTypeParmDecl*> {
  template <class BasicReader, class... Args>
  static TemplateTypeParmDecl* read(BasicReader &R, Args &&... args) {
    return R.readTemplateTypeParmDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TypeCoupledDeclRefInfo> {
  template <class BasicReader, class... Args>
  static TypeCoupledDeclRefInfo read(BasicReader &R, Args &&... args) {
    return R.readTypeCoupledDeclRefInfo(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<TypeOfKind> {
  template <class BasicReader, class... Args>
  static TypeOfKind read(BasicReader &R, Args &&... args) {
    return R.readTypeOfKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<uint32_t> {
  template <class BasicReader, class... Args>
  static uint32_t read(BasicReader &R, Args &&... args) {
    return R.readUInt32(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<uint64_t> {
  template <class BasicReader, class... Args>
  static uint64_t read(BasicReader &R, Args &&... args) {
    return R.readUInt64(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<UnaryTransformType::UTTKind> {
  template <class BasicReader, class... Args>
  static UnaryTransformType::UTTKind read(BasicReader &R, Args &&... args) {
    return R.readUnaryTypeTransformKind(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<UnsignedOrNone> {
  template <class BasicReader, class... Args>
  static UnsignedOrNone read(BasicReader &R, Args &&... args) {
    return R.readUnsignedOrNone(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<UsingShadowDecl*> {
  template <class BasicReader, class... Args>
  static UsingShadowDecl* read(BasicReader &R, Args &&... args) {
    return R.readUsingShadowDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<ValueDecl*> {
  template <class BasicReader, class... Args>
  static ValueDecl* read(BasicReader &R, Args &&... args) {
    return R.readValueDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct ReadDispatcher<VectorKind> {
  template <class BasicReader, class... Args>
  static VectorKind read(BasicReader &R, Args &&... args) {
    return R.readVectorKind(std::forward<Args>(args)...);
  }
};
template <class T>
struct ReadDispatcher<llvm::ArrayRef<T>> {
  template <class BasicReader, class... Args>
  static llvm::ArrayRef<T> read(BasicReader &R, Args &&... args) {
    return R.readArray(std::forward<Args>(args)...);
  }
};
template <class T>
struct ReadDispatcher<std::optional<T>> {
  template <class BasicReader, class... Args>
  static std::optional<T> read(BasicReader &R, Args &&... args) {
    return R.readOptional(std::forward<Args>(args)...);
  }
};

template <class ValueType>
struct UnpackOptionalValue;
template <>
struct UnpackOptionalValue<CXXRecordDecl*> {
  static std::optional<CXXRecordDecl*> unpack(CXXRecordDecl* value) {
    return value ? std::optional<CXXRecordDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<ConceptDecl*> {
  static std::optional<ConceptDecl*> unpack(ConceptDecl* value) {
    return value ? std::optional<ConceptDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<Decl*> {
  static std::optional<Decl*> unpack(Decl* value) {
    return value ? std::optional<Decl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<Expr*> {
  static std::optional<Expr*> unpack(Expr* value) {
    return value ? std::optional<Expr*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<FunctionDecl*> {
  static std::optional<FunctionDecl*> unpack(FunctionDecl* value) {
    return value ? std::optional<FunctionDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<IdentifierInfo*> {
  static std::optional<IdentifierInfo*> unpack(IdentifierInfo* value) {
    return value ? std::optional<IdentifierInfo*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<NamedDecl*> {
  static std::optional<NamedDecl*> unpack(NamedDecl* value) {
    return value ? std::optional<NamedDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<NamespaceAliasDecl*> {
  static std::optional<NamespaceAliasDecl*> unpack(NamespaceAliasDecl* value) {
    return value ? std::optional<NamespaceAliasDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<NamespaceDecl*> {
  static std::optional<NamespaceDecl*> unpack(NamespaceDecl* value) {
    return value ? std::optional<NamespaceDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<ObjCProtocolDecl*> {
  static std::optional<ObjCProtocolDecl*> unpack(ObjCProtocolDecl* value) {
    return value ? std::optional<ObjCProtocolDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<ObjCTypeParamDecl*> {
  static std::optional<ObjCTypeParamDecl*> unpack(ObjCTypeParamDecl* value) {
    return value ? std::optional<ObjCTypeParamDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<QualType> {
  static std::optional<QualType> unpack(QualType value) {
    return value.isNull() ? std::nullopt : std::optional<QualType>(value);
  }
};
template <>
struct UnpackOptionalValue<Stmt*> {
  static std::optional<Stmt*> unpack(Stmt* value) {
    return value ? std::optional<Stmt*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<TagDecl*> {
  static std::optional<TagDecl*> unpack(TagDecl* value) {
    return value ? std::optional<TagDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<TemplateDecl*> {
  static std::optional<TemplateDecl*> unpack(TemplateDecl* value) {
    return value ? std::optional<TemplateDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<TemplateName> {
  static std::optional<TemplateName> unpack(TemplateName value) {
    return value.isNull() ? std::nullopt : std::optional<TemplateName>(value);
  }
};
template <>
struct UnpackOptionalValue<TemplateTemplateParmDecl*> {
  static std::optional<TemplateTemplateParmDecl*> unpack(TemplateTemplateParmDecl* value) {
    return value ? std::optional<TemplateTemplateParmDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<TemplateTypeParmDecl*> {
  static std::optional<TemplateTypeParmDecl*> unpack(TemplateTypeParmDecl* value) {
    return value ? std::optional<TemplateTypeParmDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<uint32_t> {
  static std::optional<uint32_t> unpack(uint32_t value) {
    return value ? std::optional<uint32_t>(value - 1) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<uint64_t> {
  static std::optional<uint64_t> unpack(uint64_t value) {
    return value ? std::optional<uint64_t>(value - 1) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<UsingShadowDecl*> {
  static std::optional<UsingShadowDecl*> unpack(UsingShadowDecl* value) {
    return value ? std::optional<UsingShadowDecl*>(value) : std::nullopt;
  }
};
template <>
struct UnpackOptionalValue<ValueDecl*> {
  static std::optional<ValueDecl*> unpack(ValueDecl* value) {
    return value ? std::optional<ValueDecl*>(value) : std::nullopt;
  }
};

template <class Impl>
class BasicReaderBase {
  ASTContext &C;
protected:
  BasicReaderBase(ASTContext &ctx) : C(ctx) {}
public:
  ASTContext &getASTContext() { return C; }
  Impl &asImpl() { return static_cast<Impl&>(*this); }
  APValue readAPValue() {
    auto &ctx = asImpl().getASTContext();
    auto &&subR = asImpl().readObject();
    APValue::ValueKind kind = subR.find("kind").readAPValueKind();
    switch (kind) {
    case APValue::None: {
     return APValue(); 
    }

    case APValue::Indeterminate: {
     return APValue::IndeterminateValue(); 
    }

    case APValue::Int: {
    llvm::APSInt value = subR.find("value").readAPSInt();
     return APValue(value); 
    }

    case APValue::Float: {
    uint32_t semantics = subR.find("semantics").readUInt32();
    llvm::APInt value = subR.find("value").readAPInt();
    
    const llvm::fltSemantics &floatSema = llvm::APFloatBase::EnumToSemantics(
        static_cast<llvm::APFloatBase::Semantics>(semantics));
    return APValue(llvm::APFloat(floatSema, value));
  
    }

    case APValue::FixedPoint: {
    llvm::FixedPointSemantics semantics = subR.find("semantics").readFixedPointSemantics();
    llvm::APSInt value = subR.find("value").readAPSInt();
    
    return APValue(llvm::APFixedPoint(std::move(value), semantics));
  
    }

    case APValue::ComplexInt: {
    llvm::APSInt real = subR.find("real").readAPSInt();
    llvm::APSInt imag = subR.find("imag").readAPSInt();
     return APValue(real, imag); 
    }

    case APValue::ComplexFloat: {
    uint32_t semantics = subR.find("semantics").readUInt32();
    llvm::APInt real = subR.find("real").readAPInt();
    llvm::APInt imag = subR.find("imag").readAPInt();
    
    const llvm::fltSemantics &sema = llvm::APFloatBase::EnumToSemantics(
        static_cast<llvm::APFloatBase::Semantics>(semantics));
    return APValue(llvm::APFloat(sema, real),
                   llvm::APFloat(sema, imag));
  
    }

    case APValue::Vector: {
    llvm::SmallVector<APValue, 8> elements_buffer_0;
    llvm::ArrayRef<APValue> elements = subR.find("elements").template readArray<APValue>(elements_buffer_0);
    
    APValue result;
    result.MakeVector();
    unsigned length = elements.size();
    (void)result.setVectorUninit(length);
    for (unsigned i = 0; i < length; i++)
      result.getVectorElt(i) = elements[i];
    return result;
  
    }

    case APValue::Array: {
    uint32_t totalLength = subR.find("totalLength").readUInt32();
    bool hasFiller = subR.find("hasFiller").readBool();
    llvm::SmallVector<APValue, 8> elements_buffer_0;
    llvm::ArrayRef<APValue> elements = subR.find("elements").template readArray<APValue>(elements_buffer_0);
    
    APValue result;
    unsigned initLength = elements.size() - (hasFiller ? 1 : 0);
    result.MakeArray(initLength, totalLength);
    for (unsigned i = 0; i < initLength; ++i)
      result.getArrayInitializedElt(i) = elements[i];
    if (hasFiller)
      result.getArrayFiller() = elements.back();
    return result;
  
    }

    case APValue::Struct: {
    llvm::SmallVector<APValue, 8> bases_buffer_0;
    llvm::ArrayRef<APValue> bases = subR.find("bases").template readArray<APValue>(bases_buffer_0);
    llvm::SmallVector<APValue, 8> fields_buffer_0;
    llvm::ArrayRef<APValue> fields = subR.find("fields").template readArray<APValue>(fields_buffer_0);
    
    APValue result;
    result.MakeStruct(bases.size(), fields.size());
    for (unsigned i = 0; i < bases.size(); ++i)
      result.getStructBase(i) = bases[i];
    for (unsigned i = 0; i < fields.size(); ++i)
      result.getStructField(i) = fields[i];
    return result;
  
    }

    case APValue::Union: {
    Decl* fieldDecl = subR.find("fieldDecl").readDeclRef();
    APValue value = subR.find("value").readAPValue();
    
    return APValue(cast<clang::FieldDecl>(fieldDecl), std::move(value));
  
    }

    case APValue::AddrLabelDiff: {
    Stmt* lhs = subR.find("lhs").readStmtRef();
    Stmt* rhs = subR.find("rhs").readStmtRef();
    
    return APValue(cast<AddrLabelExpr>(lhs), cast<AddrLabelExpr>(rhs));
  
    }

    case APValue::MemberPointer: {
    bool isDerived = subR.find("isDerived").readBool();
    ValueDecl* member = subR.find("member").readValueDeclRef();
    llvm::SmallVector<CXXRecordDecl*, 8> memberPath_buffer_0;
    llvm::ArrayRef<CXXRecordDecl*> memberPath = subR.find("memberPath").template readArray<CXXRecordDecl*>(memberPath_buffer_0);
    
    APValue result;
    unsigned pathSize = memberPath.size();
    const CXXRecordDecl **pathArray =
        result.setMemberPointerUninit(member, isDerived, pathSize).data();
    for (unsigned i = 0; i < pathSize; ++i)
      pathArray[i] = memberPath[i]->getCanonicalDecl();
    return result;
  
    }

    case APValue::LValue: {
    bool hasLValuePath = subR.find("hasLValuePath").readBool();
    bool isLValueOnePastTheEnd = subR.find("isLValueOnePastTheEnd").readBool();
    bool isExpr = subR.find("isExpr").readBool();
    bool isTypeInfo = subR.find("isTypeInfo").readBool();
    bool isDynamicAlloc = subR.find("isDynamicAlloc").readBool();
    bool hasBase = subR.find("hasBase").readBool();
    bool isNullPtr = subR.find("isNullPtr").readBool();
    std::optional<QualType> typeInfo;
    if ( hasBase && isTypeInfo ) {
      typeInfo.emplace(subR.find("typeInfo").readQualType());
    }
    std::optional<uint32_t> dynamicAlloc;
    if ( hasBase && isDynamicAlloc ) {
      dynamicAlloc.emplace(subR.find("dynamicAlloc").readUInt32());
    }
    std::optional<QualType> type;
    if ( hasBase && (isTypeInfo || isDynamicAlloc) ) {
      type.emplace(subR.find("type").readQualType());
    }
    std::optional<uint32_t> callIndex;
    if ( hasBase && !isTypeInfo ) {
      callIndex.emplace(subR.find("callIndex").readUInt32());
    }
    std::optional<uint32_t> version;
    if ( hasBase && !isTypeInfo ) {
      version.emplace(subR.find("version").readUInt32());
    }
    std::optional<Stmt*> stmt;
    if ( hasBase && !isTypeInfo && isExpr ) {
      stmt.emplace(subR.find("stmt").readStmtRef());
    }
    std::optional<Decl*> decl;
    if ( hasBase && !isTypeInfo && !isDynamicAlloc && !isExpr ) {
      decl.emplace(subR.find("decl").readDeclRef());
    }
    uint32_t offsetQuantity = subR.find("offsetQuantity").readUInt32();
    llvm::SmallVector<APValue::LValuePathEntry, 8> lvaluePath_buffer_0;
    std::optional<APValue::LValuePathSerializationHelper> lvaluePath;
    if ( hasLValuePath ) {
      lvaluePath.emplace(subR.find("lvaluePath").readLValuePathSerializationHelper(lvaluePath_buffer_0));
    }
    
    (void)ctx;
    APValue::LValueBase base;
    if (hasBase) {
      if (isTypeInfo) {
        base = APValue::LValueBase::getTypeInfo(
            TypeInfoLValue(typeInfo->getTypePtr()), *type);
      } else if (isDynamicAlloc) {
        base = APValue::LValueBase::getDynamicAlloc(
            DynamicAllocLValue(*dynamicAlloc), *type);
      } else if (isExpr) {
        base = APValue::LValueBase(cast<Expr>(*stmt),
                                   *callIndex, *version);
      } else {
        base = APValue::LValueBase(cast<ValueDecl>(*decl),
                                   *callIndex, *version);
      }
    }
    CharUnits offset = CharUnits::fromQuantity(offsetQuantity);
    APValue result;
    result.MakeLValue();
    if (!hasLValuePath) {
      result.setLValue(base, offset, APValue::NoLValuePath{}, isNullPtr);
      return result;
    }
    auto pathLength = lvaluePath->Path.size();
    APValue::LValuePathEntry *path = result.setLValueUninit(
        base, offset, pathLength, isLValueOnePastTheEnd, isNullPtr).data();
    llvm::copy(lvaluePath->Path, path);
    return result;
  
    }

    }
    llvm_unreachable("bad APValue::ValueKind");
  }
  APValue::ValueKind readAPValueKind() {
    return asImpl().template readEnum<APValue::ValueKind>();
  }
  ArraySizeModifier readArraySizeModifier() {
    return asImpl().template readEnum<ArraySizeModifier>();
  }
  attr::Kind readAttrKind() {
    return asImpl().template readEnum<attr::Kind>();
  }
  AutoTypeKeyword readAutoTypeKeyword() {
    return asImpl().template readEnum<AutoTypeKeyword>();
  }
  BuiltinType::Kind readBuiltinTypeKind() {
    return asImpl().template readEnum<BuiltinType::Kind>();
  }
  CXXRecordDecl* readCXXRecordDeclRef() {
    return cast_or_null<CXXRecordDecl>(asImpl().readDeclRef());
  }
  CallingConv readCallingConv() {
    return asImpl().template readEnum<CallingConv>();
  }
  ConceptDecl* readConceptDeclRef() {
    return cast_or_null<ConceptDecl>(asImpl().readDeclRef());
  }
  DeclarationName readDeclarationName() {
    auto &ctx = asImpl().getASTContext();
    auto &&subR = asImpl().readObject();
    DeclarationName::NameKind kind = subR.find("kind").readDeclarationNameKind();
    switch (kind) {
    case DeclarationName::Identifier: {
    IdentifierInfo* identifier = subR.find("identifier").readIdentifier();
    
    return DeclarationName(identifier);
  
    }

    case DeclarationName::ObjCZeroArgSelector: {
    Selector selector = subR.find("selector").readSelector();
    
      return DeclarationName(selector);
    
    }

    case DeclarationName::ObjCOneArgSelector: {
    Selector selector = subR.find("selector").readSelector();
    
      return DeclarationName(selector);
    
    }

    case DeclarationName::ObjCMultiArgSelector: {
    Selector selector = subR.find("selector").readSelector();
    
      return DeclarationName(selector);
    
    }

    case DeclarationName::CXXConstructorName: {
    QualType type = subR.find("type").readQualType();
    
      return ctx.DeclarationNames.getCXXConstructorName(
               ctx.getCanonicalType(type));
    
    }

    case DeclarationName::CXXDestructorName: {
    QualType type = subR.find("type").readQualType();
    
      return ctx.DeclarationNames.getCXXDestructorName(
               ctx.getCanonicalType(type));
    
    }

    case DeclarationName::CXXConversionFunctionName: {
    QualType type = subR.find("type").readQualType();
    
      return ctx.DeclarationNames.getCXXConversionFunctionName(
               ctx.getCanonicalType(type));
    
    }

    case DeclarationName::CXXDeductionGuideName: {
    TemplateDecl* declaration = subR.find("declaration").readTemplateDeclRef();
    
    return ctx.DeclarationNames.getCXXDeductionGuideName(declaration);
  
    }

    case DeclarationName::CXXOperatorName: {
    OverloadedOperatorKind operatorKind = subR.find("operatorKind").readOverloadedOperatorKind();
    
    return ctx.DeclarationNames.getCXXOperatorName(operatorKind);
  
    }

    case DeclarationName::CXXLiteralOperatorName: {
    IdentifierInfo* identifier = subR.find("identifier").readIdentifier();
    
    return ctx.DeclarationNames.getCXXLiteralOperatorName(identifier);
  
    }

    case DeclarationName::CXXUsingDirective: {
    
    return DeclarationName::getUsingDirectiveName();
  
    }

    }
    llvm_unreachable("bad DeclarationName::NameKind");
  }
  DeclarationName::NameKind readDeclarationNameKind() {
    return asImpl().template readEnum<DeclarationName::NameKind>();
  }
  ElaboratedTypeKeyword readElaboratedTypeKeyword() {
    return asImpl().template readEnum<ElaboratedTypeKeyword>();
  }
  Expr* readExprRef() {
    return cast_or_null<Expr>(asImpl().readStmtRef());
  }
  FunctionDecl* readFunctionDeclRef() {
    return cast_or_null<FunctionDecl>(asImpl().readDeclRef());
  }
  NamedDecl* readNamedDeclRef() {
    return cast_or_null<NamedDecl>(asImpl().readDeclRef());
  }
  NamespaceAliasDecl* readNamespaceAliasDeclRef() {
    return cast_or_null<NamespaceAliasDecl>(asImpl().readDeclRef());
  }
  NamespaceDecl* readNamespaceDeclRef() {
    return cast_or_null<NamespaceDecl>(asImpl().readDeclRef());
  }
  NestedNameSpecifier::SpecifierKind readNestedNameSpecifierKind() {
    return asImpl().template readEnum<NestedNameSpecifier::SpecifierKind>();
  }
  ObjCProtocolDecl* readObjCProtocolDeclRef() {
    return cast_or_null<ObjCProtocolDecl>(asImpl().readDeclRef());
  }
  ObjCTypeParamDecl* readObjCTypeParamDeclRef() {
    return cast_or_null<ObjCTypeParamDecl>(asImpl().readDeclRef());
  }
  OverloadedOperatorKind readOverloadedOperatorKind() {
    return asImpl().template readEnum<OverloadedOperatorKind>();
  }
  RefQualifierKind readRefQualifierKind() {
    return asImpl().template readEnum<RefQualifierKind>();
  }
  TagDecl* readTagDeclRef() {
    return cast_or_null<TagDecl>(asImpl().readDeclRef());
  }
  TemplateArgument readTemplateArgument() {
    auto &ctx = asImpl().getASTContext();
    auto &&subR = asImpl().readObject();
    TemplateArgument::ArgKind kind = subR.find("kind").readTemplateArgumentKind();
    switch (kind) {
    case TemplateArgument::Null: {
    
    return TemplateArgument();
  
    }

    case TemplateArgument::Type: {
    QualType type = subR.find("type").readQualType();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(type, /* isNullPtr */ false, isDefaulted);
  
    }

    case TemplateArgument::Declaration: {
    ValueDecl* declaration = subR.find("declaration").readValueDeclRef();
    QualType parameterType = subR.find("parameterType").readQualType();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(declaration, parameterType, isDefaulted);
  
    }

    case TemplateArgument::NullPtr: {
    QualType type = subR.find("type").readQualType();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(type, /*nullptr*/ true, isDefaulted);
  
    }

    case TemplateArgument::Integral: {
    llvm::APSInt value = subR.find("value").readAPSInt();
    QualType type = subR.find("type").readQualType();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(ctx, value, type, isDefaulted);
  
    }

    case TemplateArgument::StructuralValue: {
    APValue value = subR.find("value").readAPValue();
    QualType type = subR.find("type").readQualType();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(ctx, type, value, isDefaulted);
  
    }

    case TemplateArgument::Template: {
    TemplateName name = subR.find("name").readTemplateName();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(name, isDefaulted);
  
    }

    case TemplateArgument::TemplateExpansion: {
    TemplateName name = subR.find("name").readTemplateName();
    UnsignedOrNone numExpansions = subR.find("numExpansions").readUnsignedOrNone();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(name, numExpansions, isDefaulted);
  
    }

    case TemplateArgument::Expression: {
    Expr* expression = subR.find("expression").readExprRef();
    bool IsCanonical = subR.find("IsCanonical").readBool();
    bool isDefaulted = subR.find("isDefaulted").readBool();
    
    return TemplateArgument(expression, IsCanonical, isDefaulted);
  
    }

    case TemplateArgument::Pack: {
    llvm::SmallVector<TemplateArgument, 8> elements_buffer_0;
    llvm::ArrayRef<TemplateArgument> elements = subR.find("elements").template readArray<TemplateArgument>(elements_buffer_0);
    
    // Copy the pack into the ASTContext.
    TemplateArgument *ctxElements = new (ctx) TemplateArgument[elements.size()];
    for (size_t i = 0, e = elements.size(); i != e; ++i)
      ctxElements[i] = elements[i];
    return TemplateArgument(llvm::ArrayRef(ctxElements, elements.size()));
  
    }

    }
    llvm_unreachable("bad TemplateArgument::ArgKind");
  }
  TemplateArgument::ArgKind readTemplateArgumentKind() {
    return asImpl().template readEnum<TemplateArgument::ArgKind>();
  }
  TemplateDecl* readTemplateDeclRef() {
    return cast_or_null<TemplateDecl>(asImpl().readDeclRef());
  }
  TemplateName readTemplateName() {
    auto &ctx = asImpl().getASTContext();
    auto &&subR = asImpl().readObject();
    TemplateName::NameKind kind = subR.find("kind").readTemplateNameKind();
    switch (kind) {
    case TemplateName::Template: {
    TemplateDecl* declaration = subR.find("declaration").readTemplateDeclRef();
    
    return TemplateName(declaration);
  
    }

    case TemplateName::UsingTemplate: {
    UsingShadowDecl* foundDecl = subR.find("foundDecl").readUsingShadowDeclRef();
    
    return TemplateName(foundDecl);
  
    }

    case TemplateName::OverloadedTemplate: {
    llvm::SmallVector<NamedDecl*, 8> overloads_buffer_0;
    llvm::ArrayRef<NamedDecl*> overloads = subR.find("overloads").template readArray<NamedDecl*>(overloads_buffer_0);
    
    // Copy into an UnresolvedSet to satisfy the interface.
    UnresolvedSet<8> overloadSet;
    for (auto overload : overloads) {
      overloadSet.addDecl(overload);
    }

    return ctx.getOverloadedTemplateName(overloadSet.begin(),
                                         overloadSet.end());
  
    }

    case TemplateName::AssumedTemplate: {
    DeclarationName name = subR.find("name").readDeclarationName();
    
    return ctx.getAssumedTemplateName(name);
  
    }

    case TemplateName::QualifiedTemplate: {
    NestedNameSpecifier * qualifier = subR.find("qualifier").readNestedNameSpecifier();
    bool hasTemplateKeyword = subR.find("hasTemplateKeyword").readBool();
    TemplateName underlyingTemplateName = subR.find("underlyingTemplateName").readTemplateName();
    
    return ctx.getQualifiedTemplateName(qualifier, hasTemplateKeyword,
                                        underlyingTemplateName);
  
    }

    case TemplateName::DependentTemplate: {
    NestedNameSpecifier * qualifier = subR.find("qualifier").readNestedNameSpecifier();
    std::optional<IdentifierInfo*> identifier = subR.find("identifier").template readOptional<IdentifierInfo*>();
    std::optional<OverloadedOperatorKind> operatorKind;
    if ( !identifier ) {
      operatorKind.emplace(subR.find("operatorKind").readOverloadedOperatorKind());
    }
    bool HasTemplateKeyword = subR.find("HasTemplateKeyword").readBool();
    
    if (identifier) {
      return ctx.getDependentTemplateName({qualifier, *identifier, HasTemplateKeyword});
    } else {
      return ctx.getDependentTemplateName({qualifier, *operatorKind, HasTemplateKeyword});
    }
  
    }

    case TemplateName::SubstTemplateTemplateParm: {
    TemplateName replacement = subR.find("replacement").readTemplateName();
    Decl* associatedDecl = subR.find("associatedDecl").readDeclRef();
    uint32_t index = subR.find("index").readUInt32();
    UnsignedOrNone packIndex = subR.find("packIndex").readUnsignedOrNone();
    bool final = subR.find("final").readBool();
    
    return ctx.getSubstTemplateTemplateParm(replacement, associatedDecl, index, packIndex, final);
  
    }

    case TemplateName::SubstTemplateTemplateParmPack: {
    TemplateArgument argumentPack = subR.find("argumentPack").readTemplateArgument();
    Decl* associatedDecl = subR.find("associatedDecl").readDeclRef();
    uint32_t index = subR.find("index").readUInt32();
    bool final = subR.find("final").readBool();
    
    return ctx.getSubstTemplateTemplateParmPack(argumentPack, associatedDecl, index, final);
  
    }

    case TemplateName::DeducedTemplate: {
    TemplateName underlying = subR.find("underlying").readTemplateName();
    uint32_t startPos = subR.find("startPos").readUInt32();
    llvm::SmallVector<TemplateArgument, 8> defaultArgs_buffer_0;
    llvm::ArrayRef<TemplateArgument> defaultArgs = subR.find("defaultArgs").template readArray<TemplateArgument>(defaultArgs_buffer_0);
    
    return ctx.getDeducedTemplateName(underlying, {startPos, defaultArgs});
  
    }

    }
    llvm_unreachable("bad TemplateName::NameKind");
  }
  TemplateName::NameKind readTemplateNameKind() {
    return asImpl().template readEnum<TemplateName::NameKind>();
  }
  TemplateTemplateParmDecl* readTemplateTemplateParmDeclRef() {
    return cast_or_null<TemplateTemplateParmDecl>(asImpl().readDeclRef());
  }
  TemplateTypeParmDecl* readTemplateTypeParmDeclRef() {
    return cast_or_null<TemplateTypeParmDecl>(asImpl().readDeclRef());
  }
  TypeOfKind readTypeOfKind() {
    return asImpl().template readEnum<TypeOfKind>();
  }
  UnaryTransformType::UTTKind readUnaryTypeTransformKind() {
    return asImpl().template readEnum<UnaryTransformType::UTTKind>();
  }
  UsingShadowDecl* readUsingShadowDeclRef() {
    return cast_or_null<UsingShadowDecl>(asImpl().readDeclRef());
  }
  ValueDecl* readValueDeclRef() {
    return cast_or_null<ValueDecl>(asImpl().readDeclRef());
  }
  VectorKind readVectorKind() {
    return asImpl().template readEnum<VectorKind>();
  }
};

