/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Helper classes for BasicWriters                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: PropertiesBase.td                                                    *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

template <class ValueType>
struct WriteDispatcher;
template <>
struct WriteDispatcher<llvm::APInt> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeAPInt(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<llvm::APSInt> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeAPSInt(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<APValue> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeAPValue(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<APValue::ValueKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeAPValueKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<ArraySizeModifier> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeArraySizeModifier(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const Attr *> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeAttr(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<attr::Kind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeAttrKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<AutoTypeKeyword> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeAutoTypeKeyword(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const BTFTypeTagAttr *> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeBTFTypeTagAttr(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<bool> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeBool(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<BuiltinType::Kind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeBuiltinTypeKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<CXXRecordDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeCXXRecordDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const CXXRecordDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeCXXRecordDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<CallingConv> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeCallingConv(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<ConceptDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeConceptDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const ConceptDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeConceptDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<Decl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const Decl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<DeclarationName> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeDeclarationName(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<DeclarationName::NameKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeDeclarationNameKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<EffectConditionExpr> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeEffectConditionExpr(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<ElaboratedTypeKeyword> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeElaboratedTypeKeyword(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<FunctionProtoType::ExceptionSpecInfo> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeExceptionSpecInfo(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<Expr*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeExprRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const Expr*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeExprRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<FunctionProtoType::ExtParameterInfo> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeExtParameterInfo(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<llvm::FixedPointSemantics> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeFixedPointSemantics(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<FunctionDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeFunctionDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const FunctionDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeFunctionDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<FunctionEffect> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeFunctionEffect(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<IdentifierInfo*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeIdentifier(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const IdentifierInfo*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeIdentifier(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<APValue::LValuePathEntry> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeLValuePathEntry(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<APValue::LValuePathSerializationHelper> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeLValuePathSerializationHelper(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<NamedDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNamedDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const NamedDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNamedDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<NamespaceAliasDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNamespaceAliasDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const NamespaceAliasDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNamespaceAliasDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<NamespaceDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNamespaceDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const NamespaceDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNamespaceDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<NestedNameSpecifier *> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNestedNameSpecifier(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<NestedNameSpecifier::SpecifierKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeNestedNameSpecifierKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<ObjCProtocolDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeObjCProtocolDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const ObjCProtocolDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeObjCProtocolDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<ObjCTypeParamDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeObjCTypeParamDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const ObjCTypeParamDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeObjCTypeParamDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<OverloadedOperatorKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeOverloadedOperatorKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<QualType> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeQualType(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<Qualifiers> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeQualifiers(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<RefQualifierKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeRefQualifierKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<Selector> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeSelector(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<SourceLocation> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeSourceLocation(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<Stmt*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeStmtRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const Stmt*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeStmtRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TagDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTagDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const TagDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTagDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TemplateArgument> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateArgument(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TemplateArgument::ArgKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateArgumentKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TemplateDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const TemplateDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TemplateName> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateName(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TemplateName::NameKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateNameKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TemplateTemplateParmDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateTemplateParmDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const TemplateTemplateParmDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateTemplateParmDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TemplateTypeParmDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateTypeParmDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const TemplateTypeParmDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTemplateTypeParmDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TypeCoupledDeclRefInfo> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTypeCoupledDeclRefInfo(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<TypeOfKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeTypeOfKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<uint32_t> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeUInt32(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<uint64_t> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeUInt64(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<UnaryTransformType::UTTKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeUnaryTypeTransformKind(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<UnsignedOrNone> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeUnsignedOrNone(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<UsingShadowDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeUsingShadowDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const UsingShadowDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeUsingShadowDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<ValueDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeValueDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<const ValueDecl*> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeValueDeclRef(std::forward<Args>(args)...);
  }
};
template <>
struct WriteDispatcher<VectorKind> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeVectorKind(std::forward<Args>(args)...);
  }
};
template <class T>
struct WriteDispatcher<llvm::ArrayRef<T>> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeArray(std::forward<Args>(args)...);
  }
};
template <class T>
struct WriteDispatcher<std::optional<T>> {
  template <class BasicWriter, class... Args>
  static void write(BasicWriter &W, Args &&... args) {
    return W.writeOptional(std::forward<Args>(args)...);
  }
};

template <class ValueType>
struct PackOptionalValue;
template <>
struct PackOptionalValue<CXXRecordDecl*> {
  static CXXRecordDecl* pack(std::optional<CXXRecordDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const CXXRecordDecl*> {
  static const CXXRecordDecl* pack(std::optional<const CXXRecordDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<ConceptDecl*> {
  static ConceptDecl* pack(std::optional<ConceptDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const ConceptDecl*> {
  static const ConceptDecl* pack(std::optional<const ConceptDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<Decl*> {
  static Decl* pack(std::optional<Decl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const Decl*> {
  static const Decl* pack(std::optional<const Decl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<Expr*> {
  static Expr* pack(std::optional<Expr*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const Expr*> {
  static const Expr* pack(std::optional<const Expr*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<FunctionDecl*> {
  static FunctionDecl* pack(std::optional<FunctionDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const FunctionDecl*> {
  static const FunctionDecl* pack(std::optional<const FunctionDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<IdentifierInfo*> {
  static IdentifierInfo* pack(std::optional<IdentifierInfo*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const IdentifierInfo*> {
  static const IdentifierInfo* pack(std::optional<const IdentifierInfo*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<NamedDecl*> {
  static NamedDecl* pack(std::optional<NamedDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const NamedDecl*> {
  static const NamedDecl* pack(std::optional<const NamedDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<NamespaceAliasDecl*> {
  static NamespaceAliasDecl* pack(std::optional<NamespaceAliasDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const NamespaceAliasDecl*> {
  static const NamespaceAliasDecl* pack(std::optional<const NamespaceAliasDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<NamespaceDecl*> {
  static NamespaceDecl* pack(std::optional<NamespaceDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const NamespaceDecl*> {
  static const NamespaceDecl* pack(std::optional<const NamespaceDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<ObjCProtocolDecl*> {
  static ObjCProtocolDecl* pack(std::optional<ObjCProtocolDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const ObjCProtocolDecl*> {
  static const ObjCProtocolDecl* pack(std::optional<const ObjCProtocolDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<ObjCTypeParamDecl*> {
  static ObjCTypeParamDecl* pack(std::optional<ObjCTypeParamDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const ObjCTypeParamDecl*> {
  static const ObjCTypeParamDecl* pack(std::optional<const ObjCTypeParamDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<QualType> {
  static QualType pack(std::optional<QualType> value) {
    return value ? *value : QualType();
  }
};
template <>
struct PackOptionalValue<Stmt*> {
  static Stmt* pack(std::optional<Stmt*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const Stmt*> {
  static const Stmt* pack(std::optional<const Stmt*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<TagDecl*> {
  static TagDecl* pack(std::optional<TagDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const TagDecl*> {
  static const TagDecl* pack(std::optional<const TagDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<TemplateDecl*> {
  static TemplateDecl* pack(std::optional<TemplateDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const TemplateDecl*> {
  static const TemplateDecl* pack(std::optional<const TemplateDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<TemplateName> {
  static TemplateName pack(std::optional<TemplateName> value) {
    return value ? *value : TemplateName();
  }
};
template <>
struct PackOptionalValue<TemplateTemplateParmDecl*> {
  static TemplateTemplateParmDecl* pack(std::optional<TemplateTemplateParmDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const TemplateTemplateParmDecl*> {
  static const TemplateTemplateParmDecl* pack(std::optional<const TemplateTemplateParmDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<TemplateTypeParmDecl*> {
  static TemplateTypeParmDecl* pack(std::optional<TemplateTypeParmDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const TemplateTypeParmDecl*> {
  static const TemplateTypeParmDecl* pack(std::optional<const TemplateTypeParmDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<uint32_t> {
  static uint32_t pack(std::optional<uint32_t> value) {
    return value ? *value + 1 : 0;
  }
};
template <>
struct PackOptionalValue<uint64_t> {
  static uint64_t pack(std::optional<uint64_t> value) {
    return value ? *value + 1 : 0;
  }
};
template <>
struct PackOptionalValue<UsingShadowDecl*> {
  static UsingShadowDecl* pack(std::optional<UsingShadowDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const UsingShadowDecl*> {
  static const UsingShadowDecl* pack(std::optional<const UsingShadowDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<ValueDecl*> {
  static ValueDecl* pack(std::optional<ValueDecl*> value) {
    return value.value_or(nullptr);
  }
};
template <>
struct PackOptionalValue<const ValueDecl*> {
  static const ValueDecl* pack(std::optional<const ValueDecl*> value) {
    return value.value_or(nullptr);
  }
};

template <class Impl>
class BasicWriterBase {
  ASTContext &C;
protected:
  BasicWriterBase(ASTContext &ctx) : C(ctx) {}
public:
  ASTContext &getASTContext() { return C; }
  Impl &asImpl() { return static_cast<Impl&>(*this); }
  void writeAPValue(const APValue & node) {
    auto &&subW = asImpl().writeObject();
    APValue::ValueKind kind = (node.getKind());
    subW.find("kind").writeAPValueKind(kind);
    switch (kind) {
    case APValue::None: {
    return;
    }

    case APValue::Indeterminate: {
    return;
    }

    case APValue::Int: {
    llvm::APSInt value = ( node.getInt() );
    subW.find("value").writeAPSInt(value);
    return;
    }

    case APValue::Float: {
    uint32_t semantics = (
      static_cast<uint32_t>(
        llvm::APFloatBase::SemanticsToEnum(node.getFloat().getSemantics()))
    );
    subW.find("semantics").writeUInt32(semantics);
    llvm::APInt value = ( node.getFloat().bitcastToAPInt() );
    subW.find("value").writeAPInt(value);
    return;
    }

    case APValue::FixedPoint: {
    llvm::FixedPointSemantics semantics = ( node.getFixedPoint().getSemantics() );
    subW.find("semantics").writeFixedPointSemantics(semantics);
    llvm::APSInt value = ( node.getFixedPoint().getValue() );
    subW.find("value").writeAPSInt(value);
    return;
    }

    case APValue::ComplexInt: {
    llvm::APSInt real = ( node.getComplexIntReal() );
    subW.find("real").writeAPSInt(real);
    llvm::APSInt imag = ( node.getComplexIntImag() );
    subW.find("imag").writeAPSInt(imag);
    return;
    }

    case APValue::ComplexFloat: {
    
    auto sema = llvm::APFloatBase::SemanticsToEnum(
        node.getComplexFloatReal().getSemantics());
    assert(sema == llvm::APFloatBase::SemanticsToEnum(
        node.getComplexFloatImag().getSemantics()));
  
    uint32_t semantics = ( static_cast<uint32_t>(sema) );
    subW.find("semantics").writeUInt32(semantics);
    llvm::APInt real = ( node.getComplexFloatReal().bitcastToAPInt() );
    subW.find("real").writeAPInt(real);
    llvm::APInt imag = ( node.getComplexFloatImag().bitcastToAPInt() );
    subW.find("imag").writeAPInt(imag);
    return;
    }

    case APValue::Vector: {
    
    SmallVector<APValue, 4> buffer;
    unsigned len = node.getVectorLength();
    for (unsigned i = 0; i < len; ++i)
      buffer.push_back(node.getVectorElt(i));
  
    llvm::ArrayRef<APValue> elements = ( buffer );
    subW.find("elements").writeArray(elements);
    return;
    }

    case APValue::Array: {
    
    SmallVector<APValue, 4> buffer{};
    unsigned initLength = node.getArrayInitializedElts();
    for (unsigned i = 0; i < initLength; ++i)
      buffer.push_back(node.getArrayInitializedElt(i));
    if (node.hasArrayFiller())
      buffer.push_back(node.getArrayFiller());
  
    uint32_t totalLength = ( node.getArraySize() );
    subW.find("totalLength").writeUInt32(totalLength);
    bool hasFiller = ( node.hasArrayFiller() );
    subW.find("hasFiller").writeBool(hasFiller);
    llvm::ArrayRef<APValue> elements = ( buffer );
    subW.find("elements").writeArray(elements);
    return;
    }

    case APValue::Struct: {
    
    SmallVector<APValue, 4> structBases;
    unsigned numBases = node.getStructNumBases();
    for (unsigned i = 0; i < numBases; ++i)
      structBases.push_back(node.getStructBase(i));
    SmallVector<APValue, 4> structFields;
    unsigned numFields = node.getStructNumFields();
    for (unsigned i = 0; i < numFields; ++i)
      structFields.push_back(node.getStructField(i));
  
    llvm::ArrayRef<APValue> bases = ( structBases );
    subW.find("bases").writeArray(bases);
    llvm::ArrayRef<APValue> fields = ( structFields );
    subW.find("fields").writeArray(fields);
    return;
    }

    case APValue::Union: {
    const Decl* fieldDecl = ( node.getUnionField() );
    subW.find("fieldDecl").writeDeclRef(fieldDecl);
    APValue value = ( node.getUnionValue() );
    subW.find("value").writeAPValue(value);
    return;
    }

    case APValue::AddrLabelDiff: {
    const Stmt* lhs = ( const_cast<AddrLabelExpr *>(node.getAddrLabelDiffLHS()) );
    subW.find("lhs").writeStmtRef(lhs);
    const Stmt* rhs = ( const_cast<AddrLabelExpr *>(node.getAddrLabelDiffRHS()) );
    subW.find("rhs").writeStmtRef(rhs);
    return;
    }

    case APValue::MemberPointer: {
    bool isDerived = ( node.isMemberPointerToDerivedMember() );
    subW.find("isDerived").writeBool(isDerived);
    const ValueDecl* member = ( node.getMemberPointerDecl() );
    subW.find("member").writeValueDeclRef(member);
    llvm::ArrayRef<const CXXRecordDecl*> memberPath = ( node.getMemberPointerPath() );
    subW.find("memberPath").writeArray(memberPath);
    return;
    }

    case APValue::LValue: {
    
    auto lvalueBase = node.getLValueBase();
    const Expr *expr =
        lvalueBase ? lvalueBase.dyn_cast<const Expr *>() : nullptr;
    bool lvalueBaseIsExpr = (bool) expr;
    bool lvalueBaseIsTypeInfo = lvalueBase.is<TypeInfoLValue>();
    bool lvalueBaseIsDynamicAlloc = lvalueBase.is<DynamicAllocLValue>();
    QualType elemTy;
    if (lvalueBase) {
      if (lvalueBaseIsTypeInfo) {
        elemTy = lvalueBase.getTypeInfoType();
      } else if (lvalueBaseIsDynamicAlloc) {
        elemTy = lvalueBase.getDynamicAllocType();
      } else if (lvalueBaseIsExpr) {
        elemTy = expr->getType();
      } else {
        elemTy = lvalueBase.get<const ValueDecl *>()->getType();
      }
    }
  
    bool hasLValuePath = ( node.hasLValuePath() );
    subW.find("hasLValuePath").writeBool(hasLValuePath);
    bool isLValueOnePastTheEnd = ( node.isLValueOnePastTheEnd() );
    subW.find("isLValueOnePastTheEnd").writeBool(isLValueOnePastTheEnd);
    bool isExpr = ( lvalueBaseIsExpr );
    subW.find("isExpr").writeBool(isExpr);
    bool isTypeInfo = ( lvalueBaseIsTypeInfo );
    subW.find("isTypeInfo").writeBool(isTypeInfo);
    bool isDynamicAlloc = ( lvalueBaseIsDynamicAlloc );
    subW.find("isDynamicAlloc").writeBool(isDynamicAlloc);
    bool hasBase = ( static_cast<bool>(lvalueBase) );
    subW.find("hasBase").writeBool(hasBase);
    bool isNullPtr = ( node.isNullPointer() );
    subW.find("isNullPtr").writeBool(isNullPtr);
    if ( hasBase && isTypeInfo ) {
    QualType typeInfo = (
      QualType(node.getLValueBase().get<TypeInfoLValue>().getType(), 0)
    );
    subW.find("typeInfo").writeQualType(typeInfo);
    }
    if ( hasBase && isDynamicAlloc ) {
    uint32_t dynamicAlloc = ( node.getLValueBase().get<DynamicAllocLValue>().getIndex() );
    subW.find("dynamicAlloc").writeUInt32(dynamicAlloc);
    }
    if ( hasBase && (isTypeInfo || isDynamicAlloc) ) {
    QualType type = (
      isTypeInfo
        ? node.getLValueBase().getTypeInfoType()
        : node.getLValueBase().getDynamicAllocType()
    );
    subW.find("type").writeQualType(type);
    }
    if ( hasBase && !isTypeInfo ) {
    uint32_t callIndex = ( node.getLValueBase().getCallIndex() );
    subW.find("callIndex").writeUInt32(callIndex);
    }
    if ( hasBase && !isTypeInfo ) {
    uint32_t version = ( node.getLValueBase().getVersion() );
    subW.find("version").writeUInt32(version);
    }
    if ( hasBase && !isTypeInfo && isExpr ) {
    const Stmt* stmt = ( const_cast<Expr *>(expr) );
    subW.find("stmt").writeStmtRef(stmt);
    }
    if ( hasBase && !isTypeInfo && !isDynamicAlloc && !isExpr ) {
    const Decl* decl = ( lvalueBase.get<const ValueDecl *>() );
    subW.find("decl").writeDeclRef(decl);
    }
    uint32_t offsetQuantity = ( node.getLValueOffset().getQuantity() );
    subW.find("offsetQuantity").writeUInt32(offsetQuantity);
    if ( hasLValuePath ) {
    APValue::LValuePathSerializationHelper lvaluePath = (
      APValue::LValuePathSerializationHelper(node.getLValuePath(), elemTy)
    );
    subW.find("lvaluePath").writeLValuePathSerializationHelper(lvaluePath);
    }
    return;
    }

    }
    llvm_unreachable("bad APValue::ValueKind");
  }
  void writeAPValueKind(APValue::ValueKind value) {
    asImpl().writeEnum(value);
  }
  void writeArraySizeModifier(ArraySizeModifier value) {
    asImpl().writeEnum(value);
  }
  void writeAttrKind(attr::Kind value) {
    asImpl().writeEnum(value);
  }
  void writeAutoTypeKeyword(AutoTypeKeyword value) {
    asImpl().writeEnum(value);
  }
  void writeBuiltinTypeKind(BuiltinType::Kind value) {
    asImpl().writeEnum(value);
  }
  void writeCXXRecordDeclRef(const CXXRecordDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeCallingConv(CallingConv value) {
    asImpl().writeEnum(value);
  }
  void writeConceptDeclRef(const ConceptDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeDeclarationName(DeclarationName node) {
    auto &&subW = asImpl().writeObject();
    DeclarationName::NameKind kind = (node.getNameKind());
    subW.find("kind").writeDeclarationNameKind(kind);
    switch (kind) {
    case DeclarationName::Identifier: {
    const IdentifierInfo* identifier = ( node.getAsIdentifierInfo() );
    subW.find("identifier").writeIdentifier(identifier);
    return;
    }

    case DeclarationName::ObjCZeroArgSelector: {
    Selector selector = ( node.getObjCSelector() );
    subW.find("selector").writeSelector(selector);
    return;
    }

    case DeclarationName::ObjCOneArgSelector: {
    Selector selector = ( node.getObjCSelector() );
    subW.find("selector").writeSelector(selector);
    return;
    }

    case DeclarationName::ObjCMultiArgSelector: {
    Selector selector = ( node.getObjCSelector() );
    subW.find("selector").writeSelector(selector);
    return;
    }

    case DeclarationName::CXXConstructorName: {
    QualType type = ( node.getCXXNameType() );
    subW.find("type").writeQualType(type);
    return;
    }

    case DeclarationName::CXXDestructorName: {
    QualType type = ( node.getCXXNameType() );
    subW.find("type").writeQualType(type);
    return;
    }

    case DeclarationName::CXXConversionFunctionName: {
    QualType type = ( node.getCXXNameType() );
    subW.find("type").writeQualType(type);
    return;
    }

    case DeclarationName::CXXDeductionGuideName: {
    const TemplateDecl* declaration = ( node.getCXXDeductionGuideTemplate() );
    subW.find("declaration").writeTemplateDeclRef(declaration);
    return;
    }

    case DeclarationName::CXXOperatorName: {
    OverloadedOperatorKind operatorKind = ( node.getCXXOverloadedOperator() );
    subW.find("operatorKind").writeOverloadedOperatorKind(operatorKind);
    return;
    }

    case DeclarationName::CXXLiteralOperatorName: {
    const IdentifierInfo* identifier = ( node.getCXXLiteralIdentifier() );
    subW.find("identifier").writeIdentifier(identifier);
    return;
    }

    case DeclarationName::CXXUsingDirective: {
    return;
    }

    }
    llvm_unreachable("bad DeclarationName::NameKind");
  }
  void writeDeclarationNameKind(DeclarationName::NameKind value) {
    asImpl().writeEnum(value);
  }
  void writeElaboratedTypeKeyword(ElaboratedTypeKeyword value) {
    asImpl().writeEnum(value);
  }
  void writeExprRef(const Expr* value) {
    asImpl().writeStmtRef(value);
  }
  void writeFunctionDeclRef(const FunctionDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeNamedDeclRef(const NamedDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeNamespaceAliasDeclRef(const NamespaceAliasDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeNamespaceDeclRef(const NamespaceDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeNestedNameSpecifierKind(NestedNameSpecifier::SpecifierKind value) {
    asImpl().writeEnum(value);
  }
  void writeObjCProtocolDeclRef(const ObjCProtocolDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeObjCTypeParamDeclRef(const ObjCTypeParamDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeOverloadedOperatorKind(OverloadedOperatorKind value) {
    asImpl().writeEnum(value);
  }
  void writeRefQualifierKind(RefQualifierKind value) {
    asImpl().writeEnum(value);
  }
  void writeTagDeclRef(const TagDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeTemplateArgument(TemplateArgument node) {
    auto &&subW = asImpl().writeObject();
    TemplateArgument::ArgKind kind = (node.getKind());
    subW.find("kind").writeTemplateArgumentKind(kind);
    switch (kind) {
    case TemplateArgument::Null: {
    return;
    }

    case TemplateArgument::Type: {
    QualType type = ( node.getAsType() );
    subW.find("type").writeQualType(type);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::Declaration: {
    const ValueDecl* declaration = ( node.getAsDecl() );
    subW.find("declaration").writeValueDeclRef(declaration);
    QualType parameterType = ( node.getParamTypeForDecl() );
    subW.find("parameterType").writeQualType(parameterType);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::NullPtr: {
    QualType type = ( node.getNullPtrType() );
    subW.find("type").writeQualType(type);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::Integral: {
    llvm::APSInt value = ( node.getAsIntegral() );
    subW.find("value").writeAPSInt(value);
    QualType type = ( node.getIntegralType() );
    subW.find("type").writeQualType(type);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::StructuralValue: {
    APValue value = ( node.getAsStructuralValue() );
    subW.find("value").writeAPValue(value);
    QualType type = ( node.getStructuralValueType() );
    subW.find("type").writeQualType(type);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::Template: {
    TemplateName name = ( node.getAsTemplateOrTemplatePattern() );
    subW.find("name").writeTemplateName(name);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::TemplateExpansion: {
    TemplateName name = ( node.getAsTemplateOrTemplatePattern() );
    subW.find("name").writeTemplateName(name);
    UnsignedOrNone numExpansions = (
      node.getNumTemplateExpansions()
    );
    subW.find("numExpansions").writeUnsignedOrNone(numExpansions);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::Expression: {
    const Expr* expression = ( node.getAsExpr() );
    subW.find("expression").writeExprRef(expression);
    bool IsCanonical = ( node.isCanonicalExpr() );
    subW.find("IsCanonical").writeBool(IsCanonical);
    bool isDefaulted = ( node.getIsDefaulted() );
    subW.find("isDefaulted").writeBool(isDefaulted);
    return;
    }

    case TemplateArgument::Pack: {
    llvm::ArrayRef<TemplateArgument> elements = ( node.pack_elements() );
    subW.find("elements").writeArray(elements);
    return;
    }

    }
    llvm_unreachable("bad TemplateArgument::ArgKind");
  }
  void writeTemplateArgumentKind(TemplateArgument::ArgKind value) {
    asImpl().writeEnum(value);
  }
  void writeTemplateDeclRef(const TemplateDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeTemplateName(TemplateName node) {
    auto &&subW = asImpl().writeObject();
    TemplateName::NameKind kind = (node.getKind());
    subW.find("kind").writeTemplateNameKind(kind);
    switch (kind) {
    case TemplateName::Template: {
    const TemplateDecl* declaration = ( node.getAsTemplateDecl() );
    subW.find("declaration").writeTemplateDeclRef(declaration);
    return;
    }

    case TemplateName::UsingTemplate: {
    const UsingShadowDecl* foundDecl = ( node.getAsUsingShadowDecl() );
    subW.find("foundDecl").writeUsingShadowDeclRef(foundDecl);
    return;
    }

    case TemplateName::OverloadedTemplate: {
    llvm::ArrayRef<const NamedDecl*> overloads = ( node.getAsOverloadedTemplate()->decls() );
    subW.find("overloads").writeArray(overloads);
    return;
    }

    case TemplateName::AssumedTemplate: {
    DeclarationName name = ( node.getAsAssumedTemplateName()->getDeclName() );
    subW.find("name").writeDeclarationName(name);
    return;
    }

    case TemplateName::QualifiedTemplate: {
    
    auto qtn = node.getAsQualifiedTemplateName();
  
    NestedNameSpecifier * qualifier = ( qtn->getQualifier() );
    subW.find("qualifier").writeNestedNameSpecifier(qualifier);
    bool hasTemplateKeyword = ( qtn->hasTemplateKeyword() );
    subW.find("hasTemplateKeyword").writeBool(hasTemplateKeyword);
    TemplateName underlyingTemplateName = ( qtn->getUnderlyingTemplate() );
    subW.find("underlyingTemplateName").writeTemplateName(underlyingTemplateName);
    return;
    }

    case TemplateName::DependentTemplate: {
    
    auto dtn = node.getAsDependentTemplateName();
    auto name = dtn->getName();
  
    NestedNameSpecifier * qualifier = ( dtn->getQualifier() );
    subW.find("qualifier").writeNestedNameSpecifier(qualifier);
    std::optional<const IdentifierInfo*> identifier = ( makeOptionalFromPointer(name.getIdentifier()) );
    subW.find("identifier").writeOptional(identifier);
    if ( !identifier ) {
    OverloadedOperatorKind operatorKind = ( name.getOperator() );
    subW.find("operatorKind").writeOverloadedOperatorKind(operatorKind);
    }
    bool HasTemplateKeyword = ( dtn->hasTemplateKeyword() );
    subW.find("HasTemplateKeyword").writeBool(HasTemplateKeyword);
    return;
    }

    case TemplateName::SubstTemplateTemplateParm: {
    
    auto parm = node.getAsSubstTemplateTemplateParm();
  
    TemplateName replacement = ( parm->getReplacement() );
    subW.find("replacement").writeTemplateName(replacement);
    const Decl* associatedDecl = ( parm->getAssociatedDecl() );
    subW.find("associatedDecl").writeDeclRef(associatedDecl);
    uint32_t index = ( parm->getIndex() );
    subW.find("index").writeUInt32(index);
    UnsignedOrNone packIndex = ( parm->getPackIndex() );
    subW.find("packIndex").writeUnsignedOrNone(packIndex);
    bool final = ( parm->getFinal() );
    subW.find("final").writeBool(final);
    return;
    }

    case TemplateName::SubstTemplateTemplateParmPack: {
    
    auto parm = node.getAsSubstTemplateTemplateParmPack();
  
    TemplateArgument argumentPack = ( parm->getArgumentPack() );
    subW.find("argumentPack").writeTemplateArgument(argumentPack);
    const Decl* associatedDecl = ( parm->getAssociatedDecl() );
    subW.find("associatedDecl").writeDeclRef(associatedDecl);
    uint32_t index = ( parm->getIndex() );
    subW.find("index").writeUInt32(index);
    bool final = ( parm->getFinal() );
    subW.find("final").writeBool(final);
    return;
    }

    case TemplateName::DeducedTemplate: {
    
    auto DTS = node.getAsDeducedTemplateName();
  
    TemplateName underlying = ( DTS->getUnderlying() );
    subW.find("underlying").writeTemplateName(underlying);
    uint32_t startPos = ( DTS->getDefaultArguments().StartPos );
    subW.find("startPos").writeUInt32(startPos);
    llvm::ArrayRef<TemplateArgument> defaultArgs = ( DTS->getDefaultArguments().Args );
    subW.find("defaultArgs").writeArray(defaultArgs);
    return;
    }

    }
    llvm_unreachable("bad TemplateName::NameKind");
  }
  void writeTemplateNameKind(TemplateName::NameKind value) {
    asImpl().writeEnum(value);
  }
  void writeTemplateTemplateParmDeclRef(const TemplateTemplateParmDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeTemplateTypeParmDeclRef(const TemplateTypeParmDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeTypeOfKind(TypeOfKind value) {
    asImpl().writeEnum(value);
  }
  void writeUnaryTypeTransformKind(UnaryTransformType::UTTKind value) {
    asImpl().writeEnum(value);
  }
  void writeUsingShadowDeclRef(const UsingShadowDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeValueDeclRef(const ValueDecl* value) {
    asImpl().writeDeclRef(value);
  }
  void writeVectorKind(VectorKind value) {
    asImpl().writeEnum(value);
  }
};

