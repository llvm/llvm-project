/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* List of AST nodes of a particular kind                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: DeclNodes.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ABSTRACT_DECL
#  define ABSTRACT_DECL(Type) Type
#endif
#ifndef DECL_RANGE
#  define DECL_RANGE(Base, First, Last)
#endif

#ifndef LAST_DECL_RANGE
#  define LAST_DECL_RANGE(Base, First, Last) DECL_RANGE(Base, First, Last)
#endif

#ifndef TRANSLATIONUNIT
#  define TRANSLATIONUNIT(Type, Base) DECL(Type, Base)
#endif
TRANSLATIONUNIT(TranslationUnit, Decl)
#undef TRANSLATIONUNIT

#ifndef TOPLEVELSTMT
#  define TOPLEVELSTMT(Type, Base) DECL(Type, Base)
#endif
TOPLEVELSTMT(TopLevelStmt, Decl)
#undef TOPLEVELSTMT

#ifndef REQUIRESEXPRBODY
#  define REQUIRESEXPRBODY(Type, Base) DECL(Type, Base)
#endif
REQUIRESEXPRBODY(RequiresExprBody, Decl)
#undef REQUIRESEXPRBODY

#ifndef OUTLINEDFUNCTION
#  define OUTLINEDFUNCTION(Type, Base) DECL(Type, Base)
#endif
OUTLINEDFUNCTION(OutlinedFunction, Decl)
#undef OUTLINEDFUNCTION

#ifndef LINKAGESPEC
#  define LINKAGESPEC(Type, Base) DECL(Type, Base)
#endif
LINKAGESPEC(LinkageSpec, Decl)
#undef LINKAGESPEC

#ifndef EXTERNCCONTEXT
#  define EXTERNCCONTEXT(Type, Base) DECL(Type, Base)
#endif
EXTERNCCONTEXT(ExternCContext, Decl)
#undef EXTERNCCONTEXT

#ifndef EXPORT
#  define EXPORT(Type, Base) DECL(Type, Base)
#endif
EXPORT(Export, Decl)
#undef EXPORT

#ifndef CAPTURED
#  define CAPTURED(Type, Base) DECL(Type, Base)
#endif
CAPTURED(Captured, Decl)
#undef CAPTURED

#ifndef BLOCK
#  define BLOCK(Type, Base) DECL(Type, Base)
#endif
BLOCK(Block, Decl)
#undef BLOCK

#ifndef STATICASSERT
#  define STATICASSERT(Type, Base) DECL(Type, Base)
#endif
STATICASSERT(StaticAssert, Decl)
#undef STATICASSERT

#ifndef PRAGMADETECTMISMATCH
#  define PRAGMADETECTMISMATCH(Type, Base) DECL(Type, Base)
#endif
PRAGMADETECTMISMATCH(PragmaDetectMismatch, Decl)
#undef PRAGMADETECTMISMATCH

#ifndef PRAGMACOMMENT
#  define PRAGMACOMMENT(Type, Base) DECL(Type, Base)
#endif
PRAGMACOMMENT(PragmaComment, Decl)
#undef PRAGMACOMMENT

#ifndef OPENACCROUTINE
#  define OPENACCROUTINE(Type, Base) DECL(Type, Base)
#endif
OPENACCROUTINE(OpenACCRoutine, Decl)
#undef OPENACCROUTINE

#ifndef OPENACCDECLARE
#  define OPENACCDECLARE(Type, Base) DECL(Type, Base)
#endif
OPENACCDECLARE(OpenACCDeclare, Decl)
#undef OPENACCDECLARE

#ifndef OBJCPROPERTYIMPL
#  define OBJCPROPERTYIMPL(Type, Base) DECL(Type, Base)
#endif
OBJCPROPERTYIMPL(ObjCPropertyImpl, Decl)
#undef OBJCPROPERTYIMPL

#ifndef OMPTHREADPRIVATE
#  define OMPTHREADPRIVATE(Type, Base) DECL(Type, Base)
#endif
OMPTHREADPRIVATE(OMPThreadPrivate, Decl)
#undef OMPTHREADPRIVATE

#ifndef OMPREQUIRES
#  define OMPREQUIRES(Type, Base) DECL(Type, Base)
#endif
OMPREQUIRES(OMPRequires, Decl)
#undef OMPREQUIRES

#ifndef OMPALLOCATE
#  define OMPALLOCATE(Type, Base) DECL(Type, Base)
#endif
OMPALLOCATE(OMPAllocate, Decl)
#undef OMPALLOCATE

#ifndef NAMED
#  define NAMED(Type, Base) DECL(Type, Base)
#endif
ABSTRACT_DECL(NAMED(Named, Decl))
#ifndef OBJCMETHOD
#  define OBJCMETHOD(Type, Base) NAMED(Type, Base)
#endif
OBJCMETHOD(ObjCMethod, NamedDecl)
#undef OBJCMETHOD

#ifndef OBJCCONTAINER
#  define OBJCCONTAINER(Type, Base) NAMED(Type, Base)
#endif
ABSTRACT_DECL(OBJCCONTAINER(ObjCContainer, NamedDecl))
#ifndef OBJCPROTOCOL
#  define OBJCPROTOCOL(Type, Base) OBJCCONTAINER(Type, Base)
#endif
OBJCPROTOCOL(ObjCProtocol, ObjCContainerDecl)
#undef OBJCPROTOCOL

#ifndef OBJCINTERFACE
#  define OBJCINTERFACE(Type, Base) OBJCCONTAINER(Type, Base)
#endif
OBJCINTERFACE(ObjCInterface, ObjCContainerDecl)
#undef OBJCINTERFACE

#ifndef OBJCIMPL
#  define OBJCIMPL(Type, Base) OBJCCONTAINER(Type, Base)
#endif
ABSTRACT_DECL(OBJCIMPL(ObjCImpl, ObjCContainerDecl))
#ifndef OBJCIMPLEMENTATION
#  define OBJCIMPLEMENTATION(Type, Base) OBJCIMPL(Type, Base)
#endif
OBJCIMPLEMENTATION(ObjCImplementation, ObjCImplDecl)
#undef OBJCIMPLEMENTATION

#ifndef OBJCCATEGORYIMPL
#  define OBJCCATEGORYIMPL(Type, Base) OBJCIMPL(Type, Base)
#endif
OBJCCATEGORYIMPL(ObjCCategoryImpl, ObjCImplDecl)
#undef OBJCCATEGORYIMPL

DECL_RANGE(ObjCImpl, ObjCImplementation, ObjCCategoryImpl)

#undef OBJCIMPL

#ifndef OBJCCATEGORY
#  define OBJCCATEGORY(Type, Base) OBJCCONTAINER(Type, Base)
#endif
OBJCCATEGORY(ObjCCategory, ObjCContainerDecl)
#undef OBJCCATEGORY

DECL_RANGE(ObjCContainer, ObjCProtocol, ObjCCategory)

#undef OBJCCONTAINER

#ifndef NAMESPACE
#  define NAMESPACE(Type, Base) NAMED(Type, Base)
#endif
NAMESPACE(Namespace, NamedDecl)
#undef NAMESPACE

#ifndef HLSLBUFFER
#  define HLSLBUFFER(Type, Base) NAMED(Type, Base)
#endif
HLSLBUFFER(HLSLBuffer, NamedDecl)
#undef HLSLBUFFER

#ifndef VALUE
#  define VALUE(Type, Base) NAMED(Type, Base)
#endif
ABSTRACT_DECL(VALUE(Value, NamedDecl))
#ifndef OMPDECLAREREDUCTION
#  define OMPDECLAREREDUCTION(Type, Base) VALUE(Type, Base)
#endif
OMPDECLAREREDUCTION(OMPDeclareReduction, ValueDecl)
#undef OMPDECLAREREDUCTION

#ifndef OMPDECLAREMAPPER
#  define OMPDECLAREMAPPER(Type, Base) VALUE(Type, Base)
#endif
OMPDECLAREMAPPER(OMPDeclareMapper, ValueDecl)
#undef OMPDECLAREMAPPER

#ifndef UNRESOLVEDUSINGVALUE
#  define UNRESOLVEDUSINGVALUE(Type, Base) VALUE(Type, Base)
#endif
UNRESOLVEDUSINGVALUE(UnresolvedUsingValue, ValueDecl)
#undef UNRESOLVEDUSINGVALUE

#ifndef UNNAMEDGLOBALCONSTANT
#  define UNNAMEDGLOBALCONSTANT(Type, Base) VALUE(Type, Base)
#endif
UNNAMEDGLOBALCONSTANT(UnnamedGlobalConstant, ValueDecl)
#undef UNNAMEDGLOBALCONSTANT

#ifndef TEMPLATEPARAMOBJECT
#  define TEMPLATEPARAMOBJECT(Type, Base) VALUE(Type, Base)
#endif
TEMPLATEPARAMOBJECT(TemplateParamObject, ValueDecl)
#undef TEMPLATEPARAMOBJECT

#ifndef MSGUID
#  define MSGUID(Type, Base) VALUE(Type, Base)
#endif
MSGUID(MSGuid, ValueDecl)
#undef MSGUID

#ifndef INDIRECTFIELD
#  define INDIRECTFIELD(Type, Base) VALUE(Type, Base)
#endif
INDIRECTFIELD(IndirectField, ValueDecl)
#undef INDIRECTFIELD

#ifndef ENUMCONSTANT
#  define ENUMCONSTANT(Type, Base) VALUE(Type, Base)
#endif
ENUMCONSTANT(EnumConstant, ValueDecl)
#undef ENUMCONSTANT

#ifndef DECLARATOR
#  define DECLARATOR(Type, Base) VALUE(Type, Base)
#endif
ABSTRACT_DECL(DECLARATOR(Declarator, ValueDecl))
#ifndef FUNCTION
#  define FUNCTION(Type, Base) DECLARATOR(Type, Base)
#endif
FUNCTION(Function, DeclaratorDecl)
#ifndef CXXMETHOD
#  define CXXMETHOD(Type, Base) FUNCTION(Type, Base)
#endif
CXXMETHOD(CXXMethod, FunctionDecl)
#ifndef CXXDESTRUCTOR
#  define CXXDESTRUCTOR(Type, Base) CXXMETHOD(Type, Base)
#endif
CXXDESTRUCTOR(CXXDestructor, CXXMethodDecl)
#undef CXXDESTRUCTOR

#ifndef CXXCONVERSION
#  define CXXCONVERSION(Type, Base) CXXMETHOD(Type, Base)
#endif
CXXCONVERSION(CXXConversion, CXXMethodDecl)
#undef CXXCONVERSION

#ifndef CXXCONSTRUCTOR
#  define CXXCONSTRUCTOR(Type, Base) CXXMETHOD(Type, Base)
#endif
CXXCONSTRUCTOR(CXXConstructor, CXXMethodDecl)
#undef CXXCONSTRUCTOR

DECL_RANGE(CXXMethod, CXXMethod, CXXConstructor)

#undef CXXMETHOD

#ifndef CXXDEDUCTIONGUIDE
#  define CXXDEDUCTIONGUIDE(Type, Base) FUNCTION(Type, Base)
#endif
CXXDEDUCTIONGUIDE(CXXDeductionGuide, FunctionDecl)
#undef CXXDEDUCTIONGUIDE

DECL_RANGE(Function, Function, CXXDeductionGuide)

#undef FUNCTION

#ifndef VAR
#  define VAR(Type, Base) DECLARATOR(Type, Base)
#endif
VAR(Var, DeclaratorDecl)
#ifndef VARTEMPLATESPECIALIZATION
#  define VARTEMPLATESPECIALIZATION(Type, Base) VAR(Type, Base)
#endif
VARTEMPLATESPECIALIZATION(VarTemplateSpecialization, VarDecl)
#ifndef VARTEMPLATEPARTIALSPECIALIZATION
#  define VARTEMPLATEPARTIALSPECIALIZATION(Type, Base) VARTEMPLATESPECIALIZATION(Type, Base)
#endif
VARTEMPLATEPARTIALSPECIALIZATION(VarTemplatePartialSpecialization, VarTemplateSpecializationDecl)
#undef VARTEMPLATEPARTIALSPECIALIZATION

DECL_RANGE(VarTemplateSpecialization, VarTemplateSpecialization, VarTemplatePartialSpecialization)

#undef VARTEMPLATESPECIALIZATION

#ifndef PARMVAR
#  define PARMVAR(Type, Base) VAR(Type, Base)
#endif
PARMVAR(ParmVar, VarDecl)
#undef PARMVAR

#ifndef OMPCAPTUREDEXPR
#  define OMPCAPTUREDEXPR(Type, Base) VAR(Type, Base)
#endif
OMPCAPTUREDEXPR(OMPCapturedExpr, VarDecl)
#undef OMPCAPTUREDEXPR

#ifndef IMPLICITPARAM
#  define IMPLICITPARAM(Type, Base) VAR(Type, Base)
#endif
IMPLICITPARAM(ImplicitParam, VarDecl)
#undef IMPLICITPARAM

#ifndef DECOMPOSITION
#  define DECOMPOSITION(Type, Base) VAR(Type, Base)
#endif
DECOMPOSITION(Decomposition, VarDecl)
#undef DECOMPOSITION

DECL_RANGE(Var, Var, Decomposition)

#undef VAR

#ifndef NONTYPETEMPLATEPARM
#  define NONTYPETEMPLATEPARM(Type, Base) DECLARATOR(Type, Base)
#endif
NONTYPETEMPLATEPARM(NonTypeTemplateParm, DeclaratorDecl)
#undef NONTYPETEMPLATEPARM

#ifndef MSPROPERTY
#  define MSPROPERTY(Type, Base) DECLARATOR(Type, Base)
#endif
MSPROPERTY(MSProperty, DeclaratorDecl)
#undef MSPROPERTY

#ifndef FIELD
#  define FIELD(Type, Base) DECLARATOR(Type, Base)
#endif
FIELD(Field, DeclaratorDecl)
#ifndef OBJCIVAR
#  define OBJCIVAR(Type, Base) FIELD(Type, Base)
#endif
OBJCIVAR(ObjCIvar, FieldDecl)
#undef OBJCIVAR

#ifndef OBJCATDEFSFIELD
#  define OBJCATDEFSFIELD(Type, Base) FIELD(Type, Base)
#endif
OBJCATDEFSFIELD(ObjCAtDefsField, FieldDecl)
#undef OBJCATDEFSFIELD

DECL_RANGE(Field, Field, ObjCAtDefsField)

#undef FIELD

DECL_RANGE(Declarator, Function, ObjCAtDefsField)

#undef DECLARATOR

#ifndef BINDING
#  define BINDING(Type, Base) VALUE(Type, Base)
#endif
BINDING(Binding, ValueDecl)
#undef BINDING

DECL_RANGE(Value, OMPDeclareReduction, Binding)

#undef VALUE

#ifndef USINGSHADOW
#  define USINGSHADOW(Type, Base) NAMED(Type, Base)
#endif
USINGSHADOW(UsingShadow, NamedDecl)
#ifndef CONSTRUCTORUSINGSHADOW
#  define CONSTRUCTORUSINGSHADOW(Type, Base) USINGSHADOW(Type, Base)
#endif
CONSTRUCTORUSINGSHADOW(ConstructorUsingShadow, UsingShadowDecl)
#undef CONSTRUCTORUSINGSHADOW

DECL_RANGE(UsingShadow, UsingShadow, ConstructorUsingShadow)

#undef USINGSHADOW

#ifndef USINGPACK
#  define USINGPACK(Type, Base) NAMED(Type, Base)
#endif
USINGPACK(UsingPack, NamedDecl)
#undef USINGPACK

#ifndef USINGDIRECTIVE
#  define USINGDIRECTIVE(Type, Base) NAMED(Type, Base)
#endif
USINGDIRECTIVE(UsingDirective, NamedDecl)
#undef USINGDIRECTIVE

#ifndef UNRESOLVEDUSINGIFEXISTS
#  define UNRESOLVEDUSINGIFEXISTS(Type, Base) NAMED(Type, Base)
#endif
UNRESOLVEDUSINGIFEXISTS(UnresolvedUsingIfExists, NamedDecl)
#undef UNRESOLVEDUSINGIFEXISTS

#ifndef TYPE
#  define TYPE(Type, Base) NAMED(Type, Base)
#endif
ABSTRACT_DECL(TYPE(Type, NamedDecl))
#ifndef TAG
#  define TAG(Type, Base) TYPE(Type, Base)
#endif
ABSTRACT_DECL(TAG(Tag, TypeDecl))
#ifndef RECORD
#  define RECORD(Type, Base) TAG(Type, Base)
#endif
RECORD(Record, TagDecl)
#ifndef CXXRECORD
#  define CXXRECORD(Type, Base) RECORD(Type, Base)
#endif
CXXRECORD(CXXRecord, RecordDecl)
#ifndef CLASSTEMPLATESPECIALIZATION
#  define CLASSTEMPLATESPECIALIZATION(Type, Base) CXXRECORD(Type, Base)
#endif
CLASSTEMPLATESPECIALIZATION(ClassTemplateSpecialization, CXXRecordDecl)
#ifndef CLASSTEMPLATEPARTIALSPECIALIZATION
#  define CLASSTEMPLATEPARTIALSPECIALIZATION(Type, Base) CLASSTEMPLATESPECIALIZATION(Type, Base)
#endif
CLASSTEMPLATEPARTIALSPECIALIZATION(ClassTemplatePartialSpecialization, ClassTemplateSpecializationDecl)
#undef CLASSTEMPLATEPARTIALSPECIALIZATION

DECL_RANGE(ClassTemplateSpecialization, ClassTemplateSpecialization, ClassTemplatePartialSpecialization)

#undef CLASSTEMPLATESPECIALIZATION

DECL_RANGE(CXXRecord, CXXRecord, ClassTemplatePartialSpecialization)

#undef CXXRECORD

DECL_RANGE(Record, Record, ClassTemplatePartialSpecialization)

#undef RECORD

#ifndef ENUM
#  define ENUM(Type, Base) TAG(Type, Base)
#endif
ENUM(Enum, TagDecl)
#undef ENUM

DECL_RANGE(Tag, Record, Enum)

#undef TAG

#ifndef UNRESOLVEDUSINGTYPENAME
#  define UNRESOLVEDUSINGTYPENAME(Type, Base) TYPE(Type, Base)
#endif
UNRESOLVEDUSINGTYPENAME(UnresolvedUsingTypename, TypeDecl)
#undef UNRESOLVEDUSINGTYPENAME

#ifndef TYPEDEFNAME
#  define TYPEDEFNAME(Type, Base) TYPE(Type, Base)
#endif
ABSTRACT_DECL(TYPEDEFNAME(TypedefName, TypeDecl))
#ifndef TYPEDEF
#  define TYPEDEF(Type, Base) TYPEDEFNAME(Type, Base)
#endif
TYPEDEF(Typedef, TypedefNameDecl)
#undef TYPEDEF

#ifndef TYPEALIAS
#  define TYPEALIAS(Type, Base) TYPEDEFNAME(Type, Base)
#endif
TYPEALIAS(TypeAlias, TypedefNameDecl)
#undef TYPEALIAS

#ifndef OBJCTYPEPARAM
#  define OBJCTYPEPARAM(Type, Base) TYPEDEFNAME(Type, Base)
#endif
OBJCTYPEPARAM(ObjCTypeParam, TypedefNameDecl)
#undef OBJCTYPEPARAM

DECL_RANGE(TypedefName, Typedef, ObjCTypeParam)

#undef TYPEDEFNAME

#ifndef TEMPLATETYPEPARM
#  define TEMPLATETYPEPARM(Type, Base) TYPE(Type, Base)
#endif
TEMPLATETYPEPARM(TemplateTypeParm, TypeDecl)
#undef TEMPLATETYPEPARM

DECL_RANGE(Type, Record, TemplateTypeParm)

#undef TYPE

#ifndef TEMPLATE
#  define TEMPLATE(Type, Base) NAMED(Type, Base)
#endif
ABSTRACT_DECL(TEMPLATE(Template, NamedDecl))
#ifndef TEMPLATETEMPLATEPARM
#  define TEMPLATETEMPLATEPARM(Type, Base) TEMPLATE(Type, Base)
#endif
TEMPLATETEMPLATEPARM(TemplateTemplateParm, TemplateDecl)
#undef TEMPLATETEMPLATEPARM

#ifndef REDECLARABLETEMPLATE
#  define REDECLARABLETEMPLATE(Type, Base) TEMPLATE(Type, Base)
#endif
ABSTRACT_DECL(REDECLARABLETEMPLATE(RedeclarableTemplate, TemplateDecl))
#ifndef VARTEMPLATE
#  define VARTEMPLATE(Type, Base) REDECLARABLETEMPLATE(Type, Base)
#endif
VARTEMPLATE(VarTemplate, RedeclarableTemplateDecl)
#undef VARTEMPLATE

#ifndef TYPEALIASTEMPLATE
#  define TYPEALIASTEMPLATE(Type, Base) REDECLARABLETEMPLATE(Type, Base)
#endif
TYPEALIASTEMPLATE(TypeAliasTemplate, RedeclarableTemplateDecl)
#undef TYPEALIASTEMPLATE

#ifndef FUNCTIONTEMPLATE
#  define FUNCTIONTEMPLATE(Type, Base) REDECLARABLETEMPLATE(Type, Base)
#endif
FUNCTIONTEMPLATE(FunctionTemplate, RedeclarableTemplateDecl)
#undef FUNCTIONTEMPLATE

#ifndef CLASSTEMPLATE
#  define CLASSTEMPLATE(Type, Base) REDECLARABLETEMPLATE(Type, Base)
#endif
CLASSTEMPLATE(ClassTemplate, RedeclarableTemplateDecl)
#undef CLASSTEMPLATE

DECL_RANGE(RedeclarableTemplate, VarTemplate, ClassTemplate)

#undef REDECLARABLETEMPLATE

#ifndef CONCEPT
#  define CONCEPT(Type, Base) TEMPLATE(Type, Base)
#endif
CONCEPT(Concept, TemplateDecl)
#undef CONCEPT

#ifndef BUILTINTEMPLATE
#  define BUILTINTEMPLATE(Type, Base) TEMPLATE(Type, Base)
#endif
BUILTINTEMPLATE(BuiltinTemplate, TemplateDecl)
#undef BUILTINTEMPLATE

DECL_RANGE(Template, TemplateTemplateParm, BuiltinTemplate)

#undef TEMPLATE

#ifndef OBJCPROPERTY
#  define OBJCPROPERTY(Type, Base) NAMED(Type, Base)
#endif
OBJCPROPERTY(ObjCProperty, NamedDecl)
#undef OBJCPROPERTY

#ifndef OBJCCOMPATIBLEALIAS
#  define OBJCCOMPATIBLEALIAS(Type, Base) NAMED(Type, Base)
#endif
OBJCCOMPATIBLEALIAS(ObjCCompatibleAlias, NamedDecl)
#undef OBJCCOMPATIBLEALIAS

#ifndef NAMESPACEALIAS
#  define NAMESPACEALIAS(Type, Base) NAMED(Type, Base)
#endif
NAMESPACEALIAS(NamespaceAlias, NamedDecl)
#undef NAMESPACEALIAS

#ifndef LABEL
#  define LABEL(Type, Base) NAMED(Type, Base)
#endif
LABEL(Label, NamedDecl)
#undef LABEL

#ifndef BASEUSING
#  define BASEUSING(Type, Base) NAMED(Type, Base)
#endif
ABSTRACT_DECL(BASEUSING(BaseUsing, NamedDecl))
#ifndef USINGENUM
#  define USINGENUM(Type, Base) BASEUSING(Type, Base)
#endif
USINGENUM(UsingEnum, BaseUsingDecl)
#undef USINGENUM

#ifndef USING
#  define USING(Type, Base) BASEUSING(Type, Base)
#endif
USING(Using, BaseUsingDecl)
#undef USING

DECL_RANGE(BaseUsing, UsingEnum, Using)

#undef BASEUSING

DECL_RANGE(Named, ObjCMethod, Using)

#undef NAMED

#ifndef LIFETIMEEXTENDEDTEMPORARY
#  define LIFETIMEEXTENDEDTEMPORARY(Type, Base) DECL(Type, Base)
#endif
LIFETIMEEXTENDEDTEMPORARY(LifetimeExtendedTemporary, Decl)
#undef LIFETIMEEXTENDEDTEMPORARY

#ifndef IMPORT
#  define IMPORT(Type, Base) DECL(Type, Base)
#endif
IMPORT(Import, Decl)
#undef IMPORT

#ifndef IMPLICITCONCEPTSPECIALIZATION
#  define IMPLICITCONCEPTSPECIALIZATION(Type, Base) DECL(Type, Base)
#endif
IMPLICITCONCEPTSPECIALIZATION(ImplicitConceptSpecialization, Decl)
#undef IMPLICITCONCEPTSPECIALIZATION

#ifndef FRIENDTEMPLATE
#  define FRIENDTEMPLATE(Type, Base) DECL(Type, Base)
#endif
FRIENDTEMPLATE(FriendTemplate, Decl)
#undef FRIENDTEMPLATE

#ifndef FRIEND
#  define FRIEND(Type, Base) DECL(Type, Base)
#endif
FRIEND(Friend, Decl)
#undef FRIEND

#ifndef FILESCOPEASM
#  define FILESCOPEASM(Type, Base) DECL(Type, Base)
#endif
FILESCOPEASM(FileScopeAsm, Decl)
#undef FILESCOPEASM

#ifndef EMPTY
#  define EMPTY(Type, Base) DECL(Type, Base)
#endif
EMPTY(Empty, Decl)
#undef EMPTY

#ifndef ACCESSSPEC
#  define ACCESSSPEC(Type, Base) DECL(Type, Base)
#endif
ACCESSSPEC(AccessSpec, Decl)
#undef ACCESSSPEC

LAST_DECL_RANGE(Decl, TranslationUnit, AccessSpec)

#undef DECL
#undef DECL_RANGE
#undef LAST_DECL_RANGE
#undef ABSTRACT_DECL
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* List of AST Decl nodes                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: DeclNodes.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef DECL_CONTEXT
#  define DECL_CONTEXT(DECL)
#endif
DECL_CONTEXT(Block)
DECL_CONTEXT(Captured)
DECL_CONTEXT(Export)
DECL_CONTEXT(ExternCContext)
DECL_CONTEXT(Function)
DECL_CONTEXT(CXXDeductionGuide)
DECL_CONTEXT(CXXMethod)
DECL_CONTEXT(CXXConstructor)
DECL_CONTEXT(CXXConversion)
DECL_CONTEXT(CXXDestructor)
DECL_CONTEXT(HLSLBuffer)
DECL_CONTEXT(LinkageSpec)
DECL_CONTEXT(Namespace)
DECL_CONTEXT(OMPDeclareMapper)
DECL_CONTEXT(OMPDeclareReduction)
DECL_CONTEXT(ObjCCategory)
DECL_CONTEXT(ObjCCategoryImpl)
DECL_CONTEXT(ObjCImplementation)
DECL_CONTEXT(ObjCInterface)
DECL_CONTEXT(ObjCProtocol)
DECL_CONTEXT(ObjCMethod)
DECL_CONTEXT(OutlinedFunction)
DECL_CONTEXT(RequiresExprBody)
DECL_CONTEXT(Enum)
DECL_CONTEXT(Record)
DECL_CONTEXT(CXXRecord)
DECL_CONTEXT(ClassTemplateSpecialization)
DECL_CONTEXT(ClassTemplatePartialSpecialization)
DECL_CONTEXT(TopLevelStmt)
DECL_CONTEXT(TranslationUnit)
#undef DECL_CONTEXT
