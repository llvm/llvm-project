//==--- Attr.td - attribute definitions -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// The documentation is organized by category. Attributes can have category-
// specific documentation that is collated within the larger document.
class DocumentationCategory<string name> {
  string Name = name;
  code Content = [{}];
}
def DocCatFunction : DocumentationCategory<"Function Attributes">;
def DocCatVariable : DocumentationCategory<"Variable Attributes">;
def DocCatType : DocumentationCategory<"Type Attributes">;
def DocCatStmt : DocumentationCategory<"Statement Attributes">;
// Attributes listed under the Undocumented category do not generate any public
// documentation. Ideally, this category should be used for internal-only
// attributes which contain no spellings.
def DocCatUndocumented : DocumentationCategory<"Undocumented">;

class DocDeprecated<string replacement = ""> {
  // If the Replacement field is empty, no replacement will be listed with the
  // documentation. Otherwise, the documentation will specify the attribute has
  // been superseded by this replacement.
  string Replacement = replacement;
}

// Specifies the documentation to be associated with the given category.
class Documentation {
  DocumentationCategory Category;
  code Content;

  // If the heading is empty, one may be picked automatically. If the attribute
  // only has one spelling, no heading is required as the attribute's sole
  // spelling is sufficient. If all spellings are semantically common, the
  // heading will be the semantic spelling. If the spellings are not
  // semantically common and no heading is provided, an error will be emitted.
  string Heading = "";

  // When set, specifies that the attribute is deprecated and can optionally
  // specify a replacement attribute.
  DocDeprecated Deprecated;
}

// Specifies that the attribute is explicitly undocumented. This can be a
// helpful placeholder for the attribute while working on the implementation,
// but should not be used once feature work has been completed.
def Undocumented : Documentation {
  let Category = DocCatUndocumented;
}

include "clang/Basic/AttrDocs.td"

// An attribute's subject is whatever it appertains to. In this file, it is
// more accurately a list of things that an attribute can appertain to. All
// Decls and Stmts are possibly AttrSubjects (even though the syntax may not
// allow attributes on a given Decl or Stmt).
class AttrSubject;

include "clang/Basic/DeclNodes.td"
include "clang/Basic/StmtNodes.td"

// A subset-subject is an AttrSubject constrained to operate only on some subset
// of that subject.
//
// The code fragment is a boolean expression that will confirm that the subject
// meets the requirements; the subject will have the name S, and will have the
// type specified by the base. It should be a simple boolean expression.
class SubsetSubject<AttrSubject base, code check> : AttrSubject {
  AttrSubject Base = base;
  code CheckCode = check;
}

// This is the type of a variable which C++11 allows alignas(...) to appertain
// to.
def NormalVar : SubsetSubject<Var,
                              [{S->getStorageClass() != VarDecl::Register &&
                                S->getKind() != Decl::ImplicitParam &&
                                S->getKind() != Decl::ParmVar &&
                                S->getKind() != Decl::NonTypeTemplateParm}]>;
def NonParmVar : SubsetSubject<Var,
                               [{S->getKind() != Decl::ParmVar}]>;
def NonBitField : SubsetSubject<Field,
                                [{!S->isBitField()}]>;

def ObjCInstanceMethod : SubsetSubject<ObjCMethod,
                                       [{S->isInstanceMethod()}]>;

def ObjCInterfaceDeclInitMethod : SubsetSubject<ObjCMethod,
                               [{S->getMethodFamily() == OMF_init &&
                                 (isa<ObjCInterfaceDecl>(S->getDeclContext()) ||
                                  (isa<ObjCCategoryDecl>(S->getDeclContext()) &&
            cast<ObjCCategoryDecl>(S->getDeclContext())->IsClassExtension()))}]>;

def Struct : SubsetSubject<Record,
                           [{!S->isUnion()}]>;

def TLSVar : SubsetSubject<Var,
                           [{S->getTLSKind() != 0}]>;

def SharedVar : SubsetSubject<Var,
                              [{S->hasGlobalStorage() && !S->getTLSKind()}]>;

def GlobalVar : SubsetSubject<Var,
                             [{S->hasGlobalStorage()}]>;

// FIXME: this hack is needed because DeclNodes.td defines the base Decl node
// type to be a class, not a definition. This makes it impossible to create an
// attribute subject which accepts a Decl. Normally, this is not a problem,
// because the attribute can have no Subjects clause to accomplish this. But in
// the case of a SubsetSubject, there's no way to express it without this hack.
def DeclBase : AttrSubject;
def FunctionLike : SubsetSubject<DeclBase,
                                  [{S->getFunctionType(false) != nullptr}]>;

def OpenCLKernelFunction : SubsetSubject<Function, [{
  S->hasAttr<OpenCLKernelAttr>()
}]>;

// HasFunctionProto is a more strict version of FunctionLike, so it should
// never be specified in a Subjects list along with FunctionLike (due to the
// inclusive nature of subject testing).
def HasFunctionProto : SubsetSubject<DeclBase,
                                     [{(S->getFunctionType(true) != nullptr &&
                              isa<FunctionProtoType>(S->getFunctionType())) ||
                                       isa<ObjCMethodDecl>(S) ||
                                       isa<BlockDecl>(S)}]>;

// A single argument to an attribute
class Argument<string name, bit optional, bit fake = 0> {
  string Name = name;
  bit Optional = optional;

  /// A fake argument is used to store and serialize additional information
  /// in an attribute without actually changing its parsing or pretty-printing.
  bit Fake = fake;
}

class BoolArgument<string name, bit opt = 0> : Argument<name, opt>;
class IdentifierArgument<string name, bit opt = 0> : Argument<name, opt>;
class IntArgument<string name, bit opt = 0> : Argument<name, opt>;
class StringArgument<string name, bit opt = 0> : Argument<name, opt>;
class ExprArgument<string name, bit opt = 0> : Argument<name, opt>;
class FunctionArgument<string name, bit opt = 0> : Argument<name, opt>;
class TypeArgument<string name, bit opt = 0> : Argument<name, opt>;
class UnsignedArgument<string name, bit opt = 0> : Argument<name, opt>;
class VariadicUnsignedArgument<string name> : Argument<name, 1>;
class VariadicExprArgument<string name> : Argument<name, 1>;
class VariadicStringArgument<string name> : Argument<name, 1>;

// A version of the form major.minor[.subminor].
class VersionArgument<string name, bit opt = 0> : Argument<name, opt>;

// This one's a doozy, so it gets its own special type
// It can be an unsigned integer, or a type. Either can
// be dependent.
class AlignedArgument<string name, bit opt = 0> : Argument<name, opt>;

// A bool argument with a default value
class DefaultBoolArgument<string name, bit default> : BoolArgument<name, 1> {
  bit Default = default;
}

// An integer argument with a default value
class DefaultIntArgument<string name, int default> : IntArgument<name, 1> {
  int Default = default;
}

// This argument is more complex, it includes the enumerator type name,
// a list of strings to accept, and a list of enumerators to map them to.
class EnumArgument<string name, string type, list<string> values,
                   list<string> enums, bit opt = 0, bit fake = 0>
    : Argument<name, opt, fake> {
  string Type = type;
  list<string> Values = values;
  list<string> Enums = enums;
}

// FIXME: There should be a VariadicArgument type that takes any other type
//        of argument and generates the appropriate type.
class VariadicEnumArgument<string name, string type, list<string> values,
                           list<string> enums> : Argument<name, 1>  {
  string Type = type;
  list<string> Values = values;
  list<string> Enums = enums;
}

// This handles one spelling of an attribute.
class Spelling<string name, string variety> {
  string Name = name;
  string Variety = variety;
  bit KnownToGCC;
}

class GNU<string name> : Spelling<name, "GNU">;
class Declspec<string name> : Spelling<name, "Declspec">;
class CXX11<string namespace, string name, int version = 1>
    : Spelling<name, "CXX11"> {
  string Namespace = namespace;
  int Version = version;
}
class Keyword<string name> : Spelling<name, "Keyword">;
class Pragma<string namespace, string name> : Spelling<name, "Pragma"> {
  string Namespace = namespace;
}

// The GCC spelling implies GNU<name, "GNU"> and CXX11<"gnu", name> and also
// sets KnownToGCC to 1. This spelling should be used for any GCC-compatible
// attributes.
class GCC<string name> : Spelling<name, "GCC"> {
  let KnownToGCC = 1;
}

class Accessor<string name, list<Spelling> spellings> {
  string Name = name;
  list<Spelling> Spellings = spellings;
}

class SubjectDiag<bit warn> {
  bit Warn = warn;
}
def WarnDiag : SubjectDiag<1>;
def ErrorDiag : SubjectDiag<0>;

class SubjectList<list<AttrSubject> subjects, SubjectDiag diag = WarnDiag,
                  string customDiag = ""> {
  list<AttrSubject> Subjects = subjects;
  SubjectDiag Diag = diag;
  string CustomDiag = customDiag;
}

class LangOpt<string name, bit negated = 0> {
  string Name = name;
  bit Negated = negated;
}
def MicrosoftExt : LangOpt<"MicrosoftExt">;
def Borland : LangOpt<"Borland">;
def CUDA : LangOpt<"CUDA">;
def COnly : LangOpt<"CPlusPlus", 1>;
def CPlusPlus : LangOpt<"CPlusPlus">;
def OpenCL : LangOpt<"OpenCL">;
def RenderScript : LangOpt<"RenderScript">;

// Defines targets for target-specific attributes. The list of strings should
// specify architectures for which the target applies, based off the ArchType
// enumeration in Triple.h.
class TargetArch<list<string> arches> {
  list<string> Arches = arches;
  list<string> OSes;
  list<string> CXXABIs;
}
def TargetARM : TargetArch<["arm", "thumb"]>;
def TargetMips : TargetArch<["mips", "mipsel"]>;
def TargetMSP430 : TargetArch<["msp430"]>;
def TargetX86 : TargetArch<["x86"]>;
def TargetAnyX86 : TargetArch<["x86", "x86_64"]>;
def TargetWindows : TargetArch<["x86", "x86_64", "arm", "thumb"]> {
  let OSes = ["Win32"];
}
def TargetMicrosoftCXXABI : TargetArch<["x86", "x86_64", "arm", "thumb"]> {
  let CXXABIs = ["Microsoft"];
}

class Attr {
  // The various ways in which an attribute can be spelled in source
  list<Spelling> Spellings;
  // The things to which an attribute can appertain
  SubjectList Subjects;
  // The arguments allowed on an attribute
  list<Argument> Args = [];
  // Accessors which should be generated for the attribute.
  list<Accessor> Accessors = [];
  // Set to true for attributes with arguments which require delayed parsing.
  bit LateParsed = 0;
  // Set to false to prevent an attribute from being propagated from a template
  // to the instantiation.
  bit Clone = 1;
  // Set to true for attributes which must be instantiated within templates
  bit TemplateDependent = 0;
  // Set to true for attributes that have a corresponding AST node.
  bit ASTNode = 1;
  // Set to true for attributes which have handler in Sema.
  bit SemaHandler = 1;
  // Set to true for attributes that are completely ignored.
  bit Ignored = 0;
  // Set to true if the attribute's parsing does not match its semantic
  // content. Eg) It parses 3 args, but semantically takes 4 args.  Opts out of
  // common attribute error checking.
  bit HasCustomParsing = 0;
  // Set to true if all of the attribute's arguments should be parsed in an
  // unevaluated context.
  bit ParseArgumentsAsUnevaluated = 0;
  // Set to true if this attribute can be duplicated on a subject when merging
  // attributes. By default, attributes are not merged.
  bit DuplicatesAllowedWhileMerging = 0;
  // Lists language options, one of which is required to be true for the
  // attribute to be applicable. If empty, no language options are required.
  list<LangOpt> LangOpts = [];
  // Any additional text that should be included verbatim in the class.
  // Note: Any additional data members will leak and should be constructed
  // externally on the ASTContext.
  code AdditionalMembers = [{}];
  // Any documentation that should be associated with the attribute. Since an
  // attribute may be documented under multiple categories, more than one
  // Documentation entry may be listed.
  list<Documentation> Documentation;
}

/// A type attribute is not processed on a declaration or a statement.
class TypeAttr : Attr {
  // By default, type attributes do not get an AST node.
  let ASTNode = 0;
}

/// A stmt attribute is not processed on a declaration or a type.
class StmtAttr : Attr;

/// An inheritable attribute is inherited by later redeclarations.
class InheritableAttr : Attr;

/// A target-specific attribute.  This class is meant to be used as a mixin
/// with InheritableAttr or Attr depending on the attribute's needs.
class TargetSpecificAttr<TargetArch target> {
  TargetArch Target = target;
  // Attributes are generally required to have unique spellings for their names
  // so that the parser can determine what kind of attribute it has parsed.
  // However, target-specific attributes are special in that the attribute only
  // "exists" for a given target. So two target-specific attributes can share
  // the same name when they exist in different targets. To support this, a
  // Kind can be explicitly specified for a target-specific attribute. This
  // corresponds to the AttributeList::AT_* enum that is generated and it
  // should contain a shared value between the attributes.
  //
  // Target-specific attributes which use this feature should ensure that the
  // spellings match exactly betweeen the attributes, and if the arguments or
  // subjects differ, should specify HasCustomParsing = 1 and implement their
  // own parsing and semantic handling requirements as-needed.
  string ParseKind;
}

/// An inheritable parameter attribute is inherited by later
/// redeclarations, even when it's written on a parameter.
class InheritableParamAttr : InheritableAttr;

/// An attribute which changes the ABI rules for a specific parameter.
class ParameterABIAttr : InheritableParamAttr {
  let Subjects = SubjectList<[ParmVar]>;
}

/// An ignored attribute, which we parse but discard with no checking.
class IgnoredAttr : Attr {
  let Ignored = 1;
  let ASTNode = 0;
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

//
// Attributes begin here
//

def AbiTag : Attr {
  let Spellings = [GCC<"abi_tag">];
  let Args = [VariadicStringArgument<"Tags">];
  let Subjects = SubjectList<[Struct, Var, Function, Namespace], ErrorDiag,
      "ExpectedStructClassVariableFunctionOrInlineNamespace">;
  let Documentation = [AbiTagsDocs];
}

def AddressSpace : TypeAttr {
  let Spellings = [GNU<"address_space">];
  let Args = [IntArgument<"AddressSpace">];
  let Documentation = [Undocumented];
}

def Alias : Attr {
  let Spellings = [GCC<"alias">];
  let Args = [StringArgument<"Aliasee">];
  let Subjects = SubjectList<[Function, GlobalVar], ErrorDiag,
                             "ExpectedFunctionGlobalVarMethodOrProperty">;
  let Documentation = [Undocumented];
}

def Aligned : InheritableAttr {
  let Spellings = [GCC<"aligned">, Declspec<"align">, Keyword<"alignas">,
                   Keyword<"_Alignas">];
//  let Subjects = SubjectList<[NonBitField, NormalVar, Tag]>;
  let Args = [AlignedArgument<"Alignment", 1>];
  let Accessors = [Accessor<"isGNU", [GCC<"aligned">]>,
                   Accessor<"isC11", [Keyword<"_Alignas">]>,
                   Accessor<"isAlignas", [Keyword<"alignas">,
                                          Keyword<"_Alignas">]>,
                   Accessor<"isDeclspec",[Declspec<"align">]>];
  let Documentation = [Undocumented];
}

def AlignValue : Attr {
  let Spellings = [
    // Unfortunately, this is semantically an assertion, not a directive
    // (something else must ensure the alignment), so aligned_value is a
    // probably a better name. We might want to add an aligned_value spelling in
    // the future (and a corresponding C++ attribute), but this can be done
    // later once we decide if we also want them to have slightly-different
    // semantics than Intel's align_value.
    GNU<"align_value">
    // Intel's compiler on Windows also supports:
    // , Declspec<"align_value">
  ];
  let Args = [ExprArgument<"Alignment">];
  let Subjects = SubjectList<[Var, TypedefName], WarnDiag,
                             "ExpectedVariableOrTypedef">;
  let Documentation = [AlignValueDocs];
}

def AlignMac68k : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def AlwaysInline : InheritableAttr {
  let Spellings = [GCC<"always_inline">, Keyword<"__forceinline">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def XRayInstrument : InheritableAttr {
  let Spellings = [GNU<"xray_always_instrument">,
                   CXX11<"clang", "xray_always_instrument">,
                   GNU<"xray_never_instrument">,
                   CXX11<"clang", "xray_never_instrument">];
  let Subjects = SubjectList<[CXXMethod, ObjCMethod, Function], WarnDiag,
                              "ExpectedFunctionOrMethod">;
  let Accessors = [Accessor<"alwaysXRayInstrument",
                     [GNU<"xray_always_instrument">,
                      CXX11<"clang", "xray_always_instrument">]>,
                   Accessor<"neverXRayInstrument",
                     [GNU<"xray_never_instrument">,
                      CXX11<"clang", "xray_never_instrument">]>];
  let Documentation = [XRayDocs];
}

def TLSModel : InheritableAttr {
  let Spellings = [GCC<"tls_model">];
  let Subjects = SubjectList<[TLSVar], ErrorDiag, "ExpectedTLSVar">;
  let Args = [StringArgument<"Model">];
  let Documentation = [TLSModelDocs];
}

def AnalyzerNoReturn : InheritableAttr {
  let Spellings = [GNU<"analyzer_noreturn">];
  let Documentation = [Undocumented];
}

def Annotate : InheritableParamAttr {
  let Spellings = [GNU<"annotate">];
  let Args = [StringArgument<"Annotation">];
  let Documentation = [Undocumented];
}

def ARMInterrupt : InheritableAttr, TargetSpecificAttr<TargetARM> {
  // NOTE: If you add any additional spellings, MSP430Interrupt's,
  // MipsInterrupt's and AnyX86Interrupt's spellings must match.
  let Spellings = [GNU<"interrupt">];
  let Args = [EnumArgument<"Interrupt", "InterruptType",
                           ["IRQ", "FIQ", "SWI", "ABORT", "UNDEF", ""],
                           ["IRQ", "FIQ", "SWI", "ABORT", "UNDEF", "Generic"],
                           1>];
  let ParseKind = "Interrupt";
  let HasCustomParsing = 1;
  let Documentation = [ARMInterruptDocs];
}

def AsmLabel : InheritableAttr {
  let Spellings = [Keyword<"asm">, Keyword<"__asm__">];
  let Args = [StringArgument<"Label">];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def Availability : InheritableAttr {
  let Spellings = [GNU<"availability">];
  let Args = [IdentifierArgument<"platform">, VersionArgument<"introduced">,
              VersionArgument<"deprecated">, VersionArgument<"obsoleted">,
              BoolArgument<"unavailable">, StringArgument<"message">,
              BoolArgument<"strict">, StringArgument<"replacement">];
  let AdditionalMembers =
[{static llvm::StringRef getPrettyPlatformName(llvm::StringRef Platform) {
    return llvm::StringSwitch<llvm::StringRef>(Platform)
             .Case("android", "Android")
             .Case("ios", "iOS")
             .Case("macos", "macOS")
             .Case("tvos", "tvOS")
             .Case("watchos", "watchOS")
             .Case("ios_app_extension", "iOS (App Extension)")
             .Case("macos_app_extension", "macOS (App Extension)")
             .Case("tvos_app_extension", "tvOS (App Extension)")
             .Case("watchos_app_extension", "watchOS (App Extension)")
             .Default(llvm::StringRef());
} }];
  let HasCustomParsing = 1;
  let DuplicatesAllowedWhileMerging = 1;
//  let Subjects = SubjectList<[Named]>;
  let Documentation = [AvailabilityDocs];
}

def Blocks : InheritableAttr {
  let Spellings = [GNU<"blocks">];
  let Args = [EnumArgument<"Type", "BlockType", ["byref"], ["ByRef"]>];
  let Documentation = [Undocumented];
}

def Bounded : IgnoredAttr {
  let Spellings = [GNU<"bounded">];
}

def CarriesDependency : InheritableParamAttr {
  let Spellings = [GNU<"carries_dependency">,
                   CXX11<"","carries_dependency", 200809>];
  let Subjects = SubjectList<[ParmVar, ObjCMethod, Function], ErrorDiag>;
  let Documentation = [CarriesDependencyDocs];
}

def CDecl : InheritableAttr {
  let Spellings = [GCC<"cdecl">, Keyword<"__cdecl">, Keyword<"_cdecl">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [Undocumented];
}

// cf_audited_transfer indicates that the given function has been
// audited and has been marked with the appropriate cf_consumed and
// cf_returns_retained attributes.  It is generally applied by
// '#pragma clang arc_cf_code_audited' rather than explicitly.
def CFAuditedTransfer : InheritableAttr {
  let Spellings = [GNU<"cf_audited_transfer">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// cf_unknown_transfer is an explicit opt-out of cf_audited_transfer.
// It indicates that the function has unknown or unautomatable
// transfer semantics.
def CFUnknownTransfer : InheritableAttr {
  let Spellings = [GNU<"cf_unknown_transfer">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def CFReturnsRetained : InheritableAttr {
  let Spellings = [GNU<"cf_returns_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
  let Documentation = [Undocumented];
}

def CFReturnsNotRetained : InheritableAttr {
  let Spellings = [GNU<"cf_returns_not_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
  let Documentation = [Undocumented];
}

def CFConsumed : InheritableParamAttr {
  let Spellings = [GNU<"cf_consumed">];
  let Subjects = SubjectList<[ParmVar]>;
  let Documentation = [Undocumented];
}

def Cleanup : InheritableAttr {
  let Spellings = [GCC<"cleanup">];
  let Args = [FunctionArgument<"FunctionDecl">];
  let Subjects = SubjectList<[Var]>;
  let Documentation = [Undocumented];
}

def Cold : InheritableAttr {
  let Spellings = [GCC<"cold">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def Common : InheritableAttr {
  let Spellings = [GCC<"common">];
  let Subjects = SubjectList<[Var]>;
  let Documentation = [Undocumented];
}

def Const : InheritableAttr {
  let Spellings = [GCC<"const">, GCC<"__const">];
  let Documentation = [Undocumented];
}

def Constructor : InheritableAttr {
  let Spellings = [GCC<"constructor">];
  let Args = [DefaultIntArgument<"Priority", 65535>];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def CUDAConstant : InheritableAttr {
  let Spellings = [GNU<"constant">];
  let Subjects = SubjectList<[Var]>;
  let LangOpts = [CUDA];
  let Documentation = [Undocumented];
}

def CUDACudartBuiltin : IgnoredAttr {
  let Spellings = [GNU<"cudart_builtin">];
  let LangOpts = [CUDA];
}

def CUDADevice : InheritableAttr {
  let Spellings = [GNU<"device">];
  let Subjects = SubjectList<[Function, Var]>;
  let LangOpts = [CUDA];
  let Documentation = [Undocumented];
}

def CUDADeviceBuiltin : IgnoredAttr {
  let Spellings = [GNU<"device_builtin">];
  let LangOpts = [CUDA];
}

def CUDADeviceBuiltinSurfaceType : IgnoredAttr {
  let Spellings = [GNU<"device_builtin_surface_type">];
  let LangOpts = [CUDA];
}

def CUDADeviceBuiltinTextureType : IgnoredAttr {
  let Spellings = [GNU<"device_builtin_texture_type">];
  let LangOpts = [CUDA];
}

def CUDAGlobal : InheritableAttr {
  let Spellings = [GNU<"global">];
  let Subjects = SubjectList<[Function]>;
  let LangOpts = [CUDA];
  let Documentation = [Undocumented];
}

def CUDAHost : InheritableAttr {
  let Spellings = [GNU<"host">];
  let Subjects = SubjectList<[Function]>;
  let LangOpts = [CUDA];
  let Documentation = [Undocumented];
}

def CUDAInvalidTarget : InheritableAttr {
  let Spellings = [];
  let Subjects = SubjectList<[Function]>;
  let LangOpts = [CUDA];
  let Documentation = [Undocumented];
}

def CUDALaunchBounds : InheritableAttr {
  let Spellings = [GNU<"launch_bounds">];
  let Args = [ExprArgument<"MaxThreads">, ExprArgument<"MinBlocks", 1>];
  let LangOpts = [CUDA];
  let Subjects = SubjectList<[ObjCMethod, FunctionLike], WarnDiag,
                             "ExpectedFunctionOrMethod">;
  // An AST node is created for this attribute, but is not used by other parts
  // of the compiler. However, this node needs to exist in the AST because
  // non-LLVM backends may be relying on the attribute's presence.
  let Documentation = [Undocumented];
}

def CUDAShared : InheritableAttr {
  let Spellings = [GNU<"shared">];
  let Subjects = SubjectList<[Var]>;
  let LangOpts = [CUDA];
  let Documentation = [Undocumented];
}

def C11NoReturn : InheritableAttr {
  let Spellings = [Keyword<"_Noreturn">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let SemaHandler = 0;
  let Documentation = [C11NoReturnDocs];
}

def CXX11NoReturn : InheritableAttr {
  let Spellings = [CXX11<"","noreturn", 200809>];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [CXX11NoReturnDocs];
}

def OpenCLKernel : InheritableAttr {
  let Spellings = [Keyword<"__kernel">, Keyword<"kernel">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def OpenCLUnrollHint : InheritableAttr {
  let Spellings = [GNU<"opencl_unroll_hint">];
  let Args = [UnsignedArgument<"UnrollHint">];
  let Documentation = [OpenCLUnrollHintDocs];
}

// This attribute is both a type attribute, and a declaration attribute (for
// parameter variables).
def OpenCLAccess : Attr {
  let Spellings = [Keyword<"__read_only">, Keyword<"read_only">,
                   Keyword<"__write_only">, Keyword<"write_only">,
                   Keyword<"__read_write">, Keyword<"read_write">];
  let Subjects = SubjectList<[ParmVar, TypedefName], ErrorDiag,
                             "ExpectedParameterOrTypedef">;
  let Accessors = [Accessor<"isReadOnly", [Keyword<"__read_only">,
                                           Keyword<"read_only">]>,
                   Accessor<"isReadWrite", [Keyword<"__read_write">,
                                            Keyword<"read_write">]>,
                   Accessor<"isWriteOnly", [Keyword<"__write_only">,
                                            Keyword<"write_only">]>];
  let Documentation = [OpenCLAccessDocs];
}

def OpenCLPrivateAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__private">, Keyword<"private">];
  let Documentation = [OpenCLAddressSpacePrivateDocs];
}

def OpenCLGlobalAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__global">, Keyword<"global">];
  let Documentation = [OpenCLAddressSpaceGlobalDocs];
}

def OpenCLLocalAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__local">, Keyword<"local">];
  let Documentation = [OpenCLAddressSpaceLocalDocs];
}

def OpenCLConstantAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__constant">, Keyword<"constant">];
  let Documentation = [OpenCLAddressSpaceConstantDocs];
}

def OpenCLGenericAddressSpace : TypeAttr {
  let Spellings = [Keyword<"__generic">, Keyword<"generic">];
  let Documentation = [OpenCLAddressSpaceGenericDocs];
}

def OpenCLNoSVM : Attr {
  let Spellings = [GNU<"nosvm">];
  let Subjects = SubjectList<[Var]>;
  let Documentation = [OpenCLNoSVMDocs];
  let LangOpts = [OpenCL];
  let ASTNode = 0;
}

def RenderScriptKernel : Attr {
  let Spellings = [GNU<"kernel">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [RenderScriptKernelAttributeDocs];
  let LangOpts = [RenderScript];
}

def Deprecated : InheritableAttr {
  let Spellings = [GCC<"deprecated">, Declspec<"deprecated">,
                   CXX11<"","deprecated", 201309>];
  let Args = [StringArgument<"Message", 1>,
              // An optional string argument that enables us to provide a
              // Fix-It.
              StringArgument<"Replacement", 1>];
  let Documentation = [DeprecatedDocs];
}

def Destructor : InheritableAttr {
  let Spellings = [GCC<"destructor">];
  let Args = [DefaultIntArgument<"Priority", 65535>];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def EmptyBases : InheritableAttr, TargetSpecificAttr<TargetMicrosoftCXXABI> {
  let Spellings = [Declspec<"empty_bases">];
  let Subjects = SubjectList<[CXXRecord]>;
  let Documentation = [EmptyBasesDocs];
}

def EnableIf : InheritableAttr {
  let Spellings = [GNU<"enable_if">];
  let Subjects = SubjectList<[Function]>;
  let Args = [ExprArgument<"Cond">, StringArgument<"Message">];
  let TemplateDependent = 1;
  let Documentation = [EnableIfDocs];
}

def ExtVectorType : Attr {
  let Spellings = [GNU<"ext_vector_type">];
  let Subjects = SubjectList<[TypedefName], ErrorDiag>;
  let Args = [ExprArgument<"NumElements">];
  let ASTNode = 0;
  let Documentation = [Undocumented];
}

def FallThrough : StmtAttr {
  let Spellings = [CXX11<"", "fallthrough", 201603>,
                   CXX11<"clang", "fallthrough">];
//  let Subjects = [NullStmt];
  let Documentation = [FallthroughDocs];
}

def FastCall : InheritableAttr {
  let Spellings = [GCC<"fastcall">, Keyword<"__fastcall">,
                   Keyword<"_fastcall">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [FastCallDocs];
}

def Final : InheritableAttr {
  let Spellings = [Keyword<"final">, Keyword<"sealed">];
  let Accessors = [Accessor<"isSpelledAsSealed", [Keyword<"sealed">]>];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def MinSize : InheritableAttr {
  let Spellings = [GNU<"minsize">];
  let Subjects = SubjectList<[Function, ObjCMethod], ErrorDiag>;
  let Documentation = [Undocumented];
}

def FlagEnum : InheritableAttr {
  let Spellings = [GNU<"flag_enum">];
  let Subjects = SubjectList<[Enum]>;
  let Documentation = [FlagEnumDocs];
  let LangOpts = [COnly];
}

def Flatten : InheritableAttr {
  let Spellings = [GCC<"flatten">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [FlattenDocs];
}

def Format : InheritableAttr {
  let Spellings = [GCC<"format">];
  let Args = [IdentifierArgument<"Type">, IntArgument<"FormatIdx">,
              IntArgument<"FirstArg">];
  let Subjects = SubjectList<[ObjCMethod, Block, HasFunctionProto], WarnDiag,
                             "ExpectedFunctionWithProtoType">;
  let Documentation = [FormatDocs];
}

def FormatArg : InheritableAttr {
  let Spellings = [GCC<"format_arg">];
  let Args = [IntArgument<"FormatIdx">];
  let Subjects = SubjectList<[ObjCMethod, HasFunctionProto], WarnDiag,
                             "ExpectedFunctionWithProtoType">;
  let Documentation = [Undocumented];
}

def GNUInline : InheritableAttr {
  let Spellings = [GCC<"gnu_inline">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def Hot : InheritableAttr {
  let Spellings = [GCC<"hot">];
  let Subjects = SubjectList<[Function]>;
  // An AST node is created for this attribute, but not actually used beyond
  // semantic checking for mutual exclusion with the Cold attribute.
  let Documentation = [Undocumented];
}

def IBAction : InheritableAttr {
  let Spellings = [GNU<"ibaction">];
  let Subjects = SubjectList<[ObjCInstanceMethod], WarnDiag,
                             "ExpectedObjCInstanceMethod">;
  // An AST node is created for this attribute, but is not used by other parts
  // of the compiler. However, this node needs to exist in the AST because
  // external tools rely on it.
  let Documentation = [Undocumented];
}

def IBOutlet : InheritableAttr {
  let Spellings = [GNU<"iboutlet">];
//  let Subjects = [ObjCIvar, ObjCProperty];
  let Documentation = [Undocumented];
}

def IBOutletCollection : InheritableAttr {
  let Spellings = [GNU<"iboutletcollection">];
  let Args = [TypeArgument<"Interface", 1>];
//  let Subjects = [ObjCIvar, ObjCProperty];
  let Documentation = [Undocumented];
}

def IFunc : Attr {
  let Spellings = [GCC<"ifunc">];
  let Args = [StringArgument<"Resolver">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [IFuncDocs];
}

def Restrict : InheritableAttr {
  let Spellings = [Declspec<"restrict">, GCC<"malloc">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def LayoutVersion : InheritableAttr, TargetSpecificAttr<TargetMicrosoftCXXABI> {
  let Spellings = [Declspec<"layout_version">];
  let Args = [UnsignedArgument<"Version">];
  let Subjects = SubjectList<[CXXRecord]>;
  let Documentation = [LayoutVersionDocs];
}

def MaxFieldAlignment : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let Args = [UnsignedArgument<"Alignment">];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def MayAlias : InheritableAttr {
  // FIXME: this is a type attribute in GCC, but a declaration attribute here.
  let Spellings = [GCC<"may_alias">];
  let Documentation = [Undocumented];
}

def MSABI : InheritableAttr {
  let Spellings = [GCC<"ms_abi">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [MSABIDocs];
}

def MSP430Interrupt : InheritableAttr, TargetSpecificAttr<TargetMSP430> {
  // NOTE: If you add any additional spellings, ARMInterrupt's, MipsInterrupt's
  // and AnyX86Interrupt's spellings must match.
  let Spellings = [GNU<"interrupt">];
  let Args = [UnsignedArgument<"Number">];
  let ParseKind = "Interrupt";
  let HasCustomParsing = 1;
  let Documentation = [Undocumented];
}

def Mips16 : InheritableAttr, TargetSpecificAttr<TargetMips> {
  let Spellings = [GCC<"mips16">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def MipsInterrupt : InheritableAttr, TargetSpecificAttr<TargetMips> {
  // NOTE: If you add any additional spellings, ARMInterrupt's,
  // MSP430Interrupt's and AnyX86Interrupt's spellings must match.
  let Spellings = [GNU<"interrupt">];
  let Subjects = SubjectList<[Function]>;
  let Args = [EnumArgument<"Interrupt", "InterruptType",
                           ["vector=sw0", "vector=sw1", "vector=hw0",
                            "vector=hw1", "vector=hw2", "vector=hw3",
                            "vector=hw4", "vector=hw5", "eic", ""],
                           ["sw0", "sw1", "hw0", "hw1", "hw2", "hw3",
                            "hw4", "hw5", "eic", "eic"]
                           >];
  let ParseKind = "Interrupt";
  let Documentation = [MipsInterruptDocs];
}

def Mode : Attr {
  let Spellings = [GCC<"mode">];
  let Subjects = SubjectList<[Var, Enum, TypedefName, Field], ErrorDiag,
                             "ExpectedVariableEnumFieldOrTypedef">;
  let Args = [IdentifierArgument<"Mode">];
  let Documentation = [Undocumented];
}

def Naked : InheritableAttr {
  let Spellings = [GCC<"naked">, Declspec<"naked">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def NeonPolyVectorType : TypeAttr {
  let Spellings = [GNU<"neon_polyvector_type">];
  let Args = [IntArgument<"NumElements">];
  let Documentation = [Undocumented];
}

def NeonVectorType : TypeAttr {
  let Spellings = [GNU<"neon_vector_type">];
  let Args = [IntArgument<"NumElements">];
  let Documentation = [Undocumented];
}

def ReturnsTwice : InheritableAttr {
  let Spellings = [GCC<"returns_twice">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def DisableTailCalls : InheritableAttr {
  let Spellings = [GNU<"disable_tail_calls">,
                   CXX11<"clang", "disable_tail_calls">];
  let Subjects = SubjectList<[Function, ObjCMethod]>;
  let Documentation = [DisableTailCallsDocs];
}

def NoAlias : InheritableAttr {
  let Spellings = [Declspec<"noalias">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [NoAliasDocs];
}

def NoCommon : InheritableAttr {
  let Spellings = [GCC<"nocommon">];
  let Subjects = SubjectList<[Var]>;
  let Documentation = [Undocumented];
}

def NoDebug : InheritableAttr {
  let Spellings = [GCC<"nodebug">];
  let Subjects = SubjectList<[FunctionLike, ObjCMethod, NonParmVar], WarnDiag,
                              "ExpectedVariableOrFunction">;
  let Documentation = [NoDebugDocs];
}

def NoDuplicate : InheritableAttr {
  let Spellings = [GNU<"noduplicate">, CXX11<"clang", "noduplicate">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [NoDuplicateDocs];
}

def NoInline : InheritableAttr {
  let Spellings = [GCC<"noinline">, Declspec<"noinline">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def NoMips16 : InheritableAttr, TargetSpecificAttr<TargetMips> {
  let Spellings = [GCC<"nomips16">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

// This is not a TargetSpecificAttr so that is silently accepted and
// ignored on other targets as encouraged by the OpenCL spec.
//
// See OpenCL 1.2 6.11.5: "It is our intention that a particular
// implementation of OpenCL be free to ignore all attributes and the
// resulting executable binary will produce the same result."
//
// However, only AMD GPU targets will emit the corresponding IR
// attribute.
//
// FIXME: This provides a sub-optimal error message if you attempt to
// use this in CUDA, since CUDA does not use the same terminology.
def AMDGPUNumVGPR : InheritableAttr {
  let Spellings = [GNU<"amdgpu_num_vgpr">];
  let Args = [UnsignedArgument<"NumVGPR">];
  let Documentation = [AMDGPUNumVGPRDocs];

// FIXME: This should be for OpenCLKernelFunction, but is not to
// workaround needing to see kernel attribute before others to know if
// this should be rejected on non-kernels.
  let Subjects = SubjectList<[Function], ErrorDiag,
                             "ExpectedKernelFunction">;
}

def AMDGPUNumSGPR : InheritableAttr {
  let Spellings = [GNU<"amdgpu_num_sgpr">];
  let Args = [UnsignedArgument<"NumSGPR">];
  let Documentation = [AMDGPUNumSGPRDocs];
  let Subjects = SubjectList<[Function], ErrorDiag,
                              "ExpectedKernelFunction">;
}

def NoSplitStack : InheritableAttr {
  let Spellings = [GCC<"no_split_stack">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [NoSplitStackDocs];
}

def NonNull : InheritableAttr {
  let Spellings = [GCC<"nonnull">];
  let Subjects = SubjectList<[ObjCMethod, HasFunctionProto, ParmVar], WarnDiag,
                             "ExpectedFunctionMethodOrParameter">;
  let Args = [VariadicUnsignedArgument<"Args">];
  let AdditionalMembers =
[{bool isNonNull(unsigned idx) const {
    if (!args_size())
      return true;
    for (const auto &V : args())
      if (V == idx)
        return true;
    return false;
  } }];
  // FIXME: We should merge duplicates into a single nonnull attribute.
  let DuplicatesAllowedWhileMerging = 1;
  let Documentation = [NonNullDocs];
}

def ReturnsNonNull : InheritableAttr {
  let Spellings = [GCC<"returns_nonnull">];
  let Subjects = SubjectList<[ObjCMethod, Function], WarnDiag,
                             "ExpectedFunctionOrMethod">;
  let Documentation = [ReturnsNonNullDocs];
}

// pass_object_size(N) indicates that the parameter should have
// __builtin_object_size with Type=N evaluated on the parameter at the callsite.
def PassObjectSize : InheritableParamAttr {
  let Spellings = [GNU<"pass_object_size">];
  let Args = [IntArgument<"Type">];
  let Subjects = SubjectList<[ParmVar]>;
  let Documentation = [PassObjectSizeDocs];
}

// Nullability type attributes.
def TypeNonNull : TypeAttr {
  let Spellings = [Keyword<"_Nonnull">];
  let Documentation = [TypeNonNullDocs];
}

def TypeNullable : TypeAttr {
  let Spellings = [Keyword<"_Nullable">];
  let Documentation = [TypeNullableDocs];
}

def TypeNullUnspecified : TypeAttr {
  let Spellings = [Keyword<"_Null_unspecified">];
  let Documentation = [TypeNullUnspecifiedDocs];
}

def ObjCKindOf : TypeAttr {
  let Spellings = [Keyword<"__kindof">];
  let Documentation = [Undocumented];
}

def AssumeAligned : InheritableAttr {
  let Spellings = [GCC<"assume_aligned">];
  let Subjects = SubjectList<[ObjCMethod, Function]>;
  let Args = [ExprArgument<"Alignment">, ExprArgument<"Offset", 1>];
  let Documentation = [AssumeAlignedDocs];
}

def NoReturn : InheritableAttr {
  let Spellings = [GCC<"noreturn">, Declspec<"noreturn">];
  // FIXME: Does GCC allow this on the function instead?
  let Documentation = [Undocumented];
}

def NoInstrumentFunction : InheritableAttr {
  let Spellings = [GCC<"no_instrument_function">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def NotTailCalled : InheritableAttr {
  let Spellings = [GNU<"not_tail_called">, CXX11<"clang", "not_tail_called">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [NotTailCalledDocs];
}

def NoThrow : InheritableAttr {
  let Spellings = [GCC<"nothrow">, Declspec<"nothrow">];
  let Documentation = [Undocumented];
}

def NvWeak : IgnoredAttr {
  let Spellings = [GNU<"nv_weak">];
  let LangOpts = [CUDA];
}

def ObjCBridge : InheritableAttr {
  let Spellings = [GNU<"objc_bridge">];
  let Subjects = SubjectList<[Record, TypedefName], ErrorDiag,
                             "ExpectedStructOrUnionOrTypedef">;
  let Args = [IdentifierArgument<"BridgedType">];
  let Documentation = [Undocumented];
}

def ObjCBridgeMutable : InheritableAttr {
  let Spellings = [GNU<"objc_bridge_mutable">];
  let Subjects = SubjectList<[Record], ErrorDiag>;
  let Args = [IdentifierArgument<"BridgedType">];
  let Documentation = [Undocumented];
}

def ObjCBridgeRelated : InheritableAttr {
  let Spellings = [GNU<"objc_bridge_related">];
  let Subjects = SubjectList<[Record], ErrorDiag>;
  let Args = [IdentifierArgument<"RelatedClass">,
          IdentifierArgument<"ClassMethod", 1>,
          IdentifierArgument<"InstanceMethod", 1>];
  let HasCustomParsing = 1;
  let Documentation = [Undocumented];
}

def NSReturnsRetained : InheritableAttr {
  let Spellings = [GNU<"ns_returns_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
  let Documentation = [Undocumented];
}

def NSReturnsNotRetained : InheritableAttr {
  let Spellings = [GNU<"ns_returns_not_retained">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
  let Documentation = [Undocumented];
}

def NSReturnsAutoreleased : InheritableAttr {
  let Spellings = [GNU<"ns_returns_autoreleased">];
//  let Subjects = SubjectList<[ObjCMethod, ObjCProperty, Function]>;
  let Documentation = [Undocumented];
}

def NSConsumesSelf : InheritableAttr {
  let Spellings = [GNU<"ns_consumes_self">];
  let Subjects = SubjectList<[ObjCMethod]>;
  let Documentation = [Undocumented];
}

def NSConsumed : InheritableParamAttr {
  let Spellings = [GNU<"ns_consumed">];
  let Subjects = SubjectList<[ParmVar]>;
  let Documentation = [Undocumented];
}

def ObjCException : InheritableAttr {
  let Spellings = [GNU<"objc_exception">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
  let Documentation = [Undocumented];
}

def ObjCMethodFamily : InheritableAttr {
  let Spellings = [GNU<"objc_method_family">];
  let Subjects = SubjectList<[ObjCMethod], ErrorDiag>;
  let Args = [EnumArgument<"Family", "FamilyKind",
               ["none", "alloc", "copy", "init", "mutableCopy", "new"],
               ["OMF_None", "OMF_alloc", "OMF_copy", "OMF_init",
                "OMF_mutableCopy", "OMF_new"]>];
  let Documentation = [ObjCMethodFamilyDocs];
}

def ObjCNSObject : InheritableAttr {
  let Spellings = [GNU<"NSObject">];
  let Documentation = [Undocumented];
}

def ObjCIndependentClass : InheritableAttr {
  let Spellings = [GNU<"objc_independent_class">];
  let Documentation = [Undocumented];
}

def ObjCPreciseLifetime : InheritableAttr {
  let Spellings = [GNU<"objc_precise_lifetime">];
  let Subjects = SubjectList<[Var], ErrorDiag>;
  let Documentation = [Undocumented];
}

def ObjCReturnsInnerPointer : InheritableAttr {
  let Spellings = [GNU<"objc_returns_inner_pointer">];
  let Subjects = SubjectList<[ObjCMethod, ObjCProperty], ErrorDiag>;
  let Documentation = [Undocumented];
}

def ObjCRequiresSuper : InheritableAttr {
  let Spellings = [GNU<"objc_requires_super">];
  let Subjects = SubjectList<[ObjCMethod], ErrorDiag>;
  let Documentation = [ObjCRequiresSuperDocs];
}

def ObjCRootClass : InheritableAttr {
  let Spellings = [GNU<"objc_root_class">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
  let Documentation = [Undocumented];
}

def ObjCExplicitProtocolImpl : InheritableAttr {
  let Spellings = [GNU<"objc_protocol_requires_explicit_implementation">];
  let Subjects = SubjectList<[ObjCProtocol], ErrorDiag>;
  let Documentation = [Undocumented];
}

def ObjCDesignatedInitializer : Attr {
  let Spellings = [GNU<"objc_designated_initializer">];
  let Subjects = SubjectList<[ObjCInterfaceDeclInitMethod], ErrorDiag,
                             "ExpectedObjCInterfaceDeclInitMethod">;
  let Documentation = [Undocumented];
}

def ObjCRuntimeName : Attr {
  let Spellings = [GNU<"objc_runtime_name">];
  let Subjects = SubjectList<[ObjCInterface, ObjCProtocol], ErrorDiag>;
  let Args = [StringArgument<"MetadataName">];
  let Documentation = [ObjCRuntimeNameDocs];
}

def ObjCRuntimeVisible : Attr {
  let Spellings = [GNU<"objc_runtime_visible">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
  let Documentation = [ObjCRuntimeVisibleDocs];
}

def ObjCBoxable : Attr {
  let Spellings = [GNU<"objc_boxable">];
  let Subjects = SubjectList<[Record], ErrorDiag, "ExpectedStructOrUnion">;
  let Documentation = [ObjCBoxableDocs];
}

def OptimizeNone : InheritableAttr {
  let Spellings = [GNU<"optnone">, CXX11<"clang", "optnone">];
  let Subjects = SubjectList<[Function, ObjCMethod]>;
  let Documentation = [OptnoneDocs];
}

def Overloadable : Attr {
  let Spellings = [GNU<"overloadable">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [OverloadableDocs];
}

def Override : InheritableAttr { 
  let Spellings = [Keyword<"override">];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def Ownership : InheritableAttr {
  let Spellings = [GNU<"ownership_holds">, GNU<"ownership_returns">,
                   GNU<"ownership_takes">];
  let Accessors = [Accessor<"isHolds", [GNU<"ownership_holds">]>,
                   Accessor<"isReturns", [GNU<"ownership_returns">]>,
                   Accessor<"isTakes", [GNU<"ownership_takes">]>];
  let AdditionalMembers = [{
    enum OwnershipKind { Holds, Returns, Takes };
    OwnershipKind getOwnKind() const {
      return isHolds() ? Holds :
             isTakes() ? Takes :
             Returns;
    }
  }];
  let Args = [IdentifierArgument<"Module">, VariadicUnsignedArgument<"Args">];
  let Subjects = SubjectList<[HasFunctionProto], WarnDiag,
                             "ExpectedFunctionWithProtoType">;
  let Documentation = [Undocumented];
}

def Packed : InheritableAttr {
  let Spellings = [GCC<"packed">];
//  let Subjects = [Tag, Field];
  let Documentation = [Undocumented];
}

def IntelOclBicc : InheritableAttr {
  let Spellings = [GNU<"intel_ocl_bicc">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [Undocumented];
}

def Pcs : InheritableAttr {
  let Spellings = [GCC<"pcs">];
  let Args = [EnumArgument<"PCS", "PCSType",
                           ["aapcs", "aapcs-vfp"],
                           ["AAPCS", "AAPCS_VFP"]>];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [PcsDocs];
}

def Pure : InheritableAttr {
  let Spellings = [GCC<"pure">];
  let Documentation = [Undocumented];
}

def Regparm : TypeAttr {
  let Spellings = [GCC<"regparm">];
  let Args = [UnsignedArgument<"NumParams">];
  let Documentation = [RegparmDocs];
}

def ReqdWorkGroupSize : InheritableAttr {
  let Spellings = [GNU<"reqd_work_group_size">];
  let Args = [UnsignedArgument<"XDim">, UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def RequireConstantInit : InheritableAttr {
  let Spellings = [GNU<"require_constant_initialization">,
                   CXX11<"clang", "require_constant_initialization">];
  let Subjects = SubjectList<[GlobalVar], ErrorDiag,
                              "ExpectedStaticOrTLSVar">;
  let Documentation = [RequireConstantInitDocs];
  let LangOpts = [CPlusPlus];
}

def WorkGroupSizeHint :  InheritableAttr {
  let Spellings = [GNU<"work_group_size_hint">];
  let Args = [UnsignedArgument<"XDim">, 
              UnsignedArgument<"YDim">,
              UnsignedArgument<"ZDim">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def InitPriority : InheritableAttr {
  let Spellings = [GNU<"init_priority">];
  let Args = [UnsignedArgument<"Priority">];
  let Subjects = SubjectList<[Var], ErrorDiag>;
  let Documentation = [Undocumented];
}

def Section : InheritableAttr {
  let Spellings = [GCC<"section">, Declspec<"allocate">];
  let Args = [StringArgument<"Name">];
  let Subjects = SubjectList<[Function, GlobalVar,
                              ObjCMethod, ObjCProperty], ErrorDiag,
                             "ExpectedFunctionGlobalVarMethodOrProperty">;
  let Documentation = [SectionDocs];
}

def Sentinel : InheritableAttr {
  let Spellings = [GCC<"sentinel">];
  let Args = [DefaultIntArgument<"Sentinel", 0>,
              DefaultIntArgument<"NullPos", 0>];
//  let Subjects = SubjectList<[Function, ObjCMethod, Block, Var]>;
  let Documentation = [Undocumented];
}

def StdCall : InheritableAttr {
  let Spellings = [GCC<"stdcall">, Keyword<"__stdcall">, Keyword<"_stdcall">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [StdCallDocs];
}

def SwiftCall : InheritableAttr {
  let Spellings = [GCC<"swiftcall">];
//  let Subjects = SubjectList<[Function]>;
  let Documentation = [SwiftCallDocs];
}

def SwiftContext : ParameterABIAttr {
  let Spellings = [GCC<"swift_context">];
  let Documentation = [SwiftContextDocs];
}

def SwiftErrorResult : ParameterABIAttr {
  let Spellings = [GCC<"swift_error_result">];
  let Documentation = [SwiftErrorResultDocs];
}

def SwiftIndirectResult : ParameterABIAttr {
  let Spellings = [GCC<"swift_indirect_result">];
  let Documentation = [SwiftIndirectResultDocs];
}

def SysVABI : InheritableAttr {
  let Spellings = [GCC<"sysv_abi">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [Undocumented];
}

def ThisCall : InheritableAttr {
  let Spellings = [GCC<"thiscall">, Keyword<"__thiscall">,
                   Keyword<"_thiscall">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [ThisCallDocs];
}

def VectorCall : InheritableAttr {
  let Spellings = [GNU<"vectorcall">, Keyword<"__vectorcall">,
                   Keyword<"_vectorcall">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [VectorCallDocs];
}

def Pascal : InheritableAttr {
  let Spellings = [GNU<"pascal">, Keyword<"__pascal">, Keyword<"_pascal">];
//  let Subjects = [Function, ObjCMethod];
  let Documentation = [Undocumented];
}

def PreserveMost : InheritableAttr {
  let Spellings = [GNU<"preserve_most">];
  let Documentation = [PreserveMostDocs];
}

def PreserveAll : InheritableAttr {
  let Spellings = [GNU<"preserve_all">];
  let Documentation = [PreserveAllDocs];
}

def Target : InheritableAttr {
  let Spellings = [GCC<"target">];
  let Args = [StringArgument<"featuresStr">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [TargetDocs];
  let AdditionalMembers = [{
    typedef std::pair<std::vector<std::string>, StringRef> ParsedTargetAttr;
    ParsedTargetAttr parse() const {
      ParsedTargetAttr Ret;
      SmallVector<StringRef, 1> AttrFeatures;
      getFeaturesStr().split(AttrFeatures, ",");

      // Grab the various features and prepend a "+" to turn on the feature to
      // the backend and add them to our existing set of features.
      for (auto &Feature : AttrFeatures) {
        // Go ahead and trim whitespace rather than either erroring or
        // accepting it weirdly.
        Feature = Feature.trim();

        // We don't support cpu tuning this way currently.
        // TODO: Support the fpmath option. It will require checking
        // overall feature validity for the function with the rest of the
        // attributes on the function.
        if (Feature.startswith("fpmath=") || Feature.startswith("tune="))
	  continue;

        // While we're here iterating check for a different target cpu.
        if (Feature.startswith("arch="))
          Ret.second = Feature.split("=").second.trim();
        else if (Feature.startswith("no-"))
          Ret.first.push_back("-" + Feature.split("-").second.str());
        else
          Ret.first.push_back("+" + Feature.str());
      }
      return Ret;
    }
  }];
}

def TransparentUnion : InheritableAttr {
  let Spellings = [GCC<"transparent_union">];
//  let Subjects = SubjectList<[Record, TypedefName]>;
  let Documentation = [Undocumented];
}

def Unavailable : InheritableAttr {
  let Spellings = [GNU<"unavailable">];
  let Args = [StringArgument<"Message", 1>,
              EnumArgument<"ImplicitReason", "ImplicitReason",
                ["", "", "", ""],
                ["IR_None",
                 "IR_ARCForbiddenType",
                 "IR_ForbiddenWeak",
                 "IR_ARCForbiddenConversion",
                 "IR_ARCInitReturnsUnrelated",
                 "IR_ARCFieldWithOwnership"], 1, /*fake*/ 1>];
  let Documentation = [Undocumented];
}

def ArcWeakrefUnavailable : InheritableAttr {
  let Spellings = [GNU<"objc_arc_weak_reference_unavailable">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
  let Documentation = [Undocumented];
}

def ObjCGC : TypeAttr {
  let Spellings = [GNU<"objc_gc">];
  let Args = [IdentifierArgument<"Kind">];
  let Documentation = [Undocumented];
}

def ObjCOwnership : InheritableAttr {
  let Spellings = [GNU<"objc_ownership">];
  let Args = [IdentifierArgument<"Kind">];
  let ASTNode = 0;
  let Documentation = [Undocumented];
}

def ObjCRequiresPropertyDefs : InheritableAttr {
  let Spellings = [GNU<"objc_requires_property_definitions">];
  let Subjects = SubjectList<[ObjCInterface], ErrorDiag>;
  let Documentation = [Undocumented];
}

def Unused : InheritableAttr {
  let Spellings = [CXX11<"", "maybe_unused", 201603>, GCC<"unused">];
  let Subjects = SubjectList<[Var, ObjCIvar, Type, Enum, EnumConstant, Label,
                              Field, ObjCMethod, FunctionLike], WarnDiag,
                             "ExpectedForMaybeUnused">;
  let Documentation = [WarnMaybeUnusedDocs];
}

def Used : InheritableAttr {
  let Spellings = [GCC<"used">];
  let Documentation = [Undocumented];
}

def Uuid : InheritableAttr {
  let Spellings = [Declspec<"uuid">];
  let Args = [StringArgument<"Guid">];
//  let Subjects = SubjectList<[CXXRecord]>;
  let LangOpts = [MicrosoftExt, Borland];
  let Documentation = [Undocumented];
}

def VectorSize : TypeAttr {
  let Spellings = [GCC<"vector_size">];
  let Args = [ExprArgument<"NumBytes">];
  let Documentation = [Undocumented];
}

def VecTypeHint : InheritableAttr {
  let Spellings = [GNU<"vec_type_hint">];
  let Args = [TypeArgument<"TypeHint">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def Visibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GCC<"visibility">];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
  let Documentation = [Undocumented];
}

def TypeVisibility : InheritableAttr {
  let Clone = 0;
  let Spellings = [GNU<"type_visibility">, CXX11<"clang", "type_visibility">];
  let Args = [EnumArgument<"Visibility", "VisibilityType",
                           ["default", "hidden", "internal", "protected"],
                           ["Default", "Hidden", "Hidden", "Protected"]>];
//  let Subjects = [Tag, ObjCInterface, Namespace];
  let Documentation = [Undocumented];
}

def VecReturn : InheritableAttr {
  let Spellings = [GNU<"vecreturn">];
  let Subjects = SubjectList<[CXXRecord], ErrorDiag>;
  let Documentation = [Undocumented];
}

def WarnUnused : InheritableAttr {
  let Spellings = [GNU<"warn_unused">];
  let Subjects = SubjectList<[Record]>;
  let Documentation = [Undocumented];
}

def WarnUnusedResult : InheritableAttr {
  let Spellings = [CXX11<"", "nodiscard", 201603>,
                   CXX11<"clang", "warn_unused_result">,
                   GCC<"warn_unused_result">];
  let Subjects = SubjectList<[ObjCMethod, Enum, CXXRecord, FunctionLike],
                             WarnDiag, "ExpectedFunctionMethodEnumOrClass">;
  let Documentation = [WarnUnusedResultsDocs];
}

def Weak : InheritableAttr {
  let Spellings = [GCC<"weak">];
  let Subjects = SubjectList<[Var, Function, CXXRecord]>;
  let Documentation = [Undocumented];
}

def WeakImport : InheritableAttr {
  let Spellings = [GNU<"weak_import">];
  let Documentation = [Undocumented];
}

def WeakRef : InheritableAttr {
  let Spellings = [GCC<"weakref">];
  // A WeakRef that has an argument is treated as being an AliasAttr
  let Args = [StringArgument<"Aliasee", 1>];
  let Subjects = SubjectList<[Var, Function], ErrorDiag>;
  let Documentation = [Undocumented];
}

def LTOVisibilityPublic : InheritableAttr {
  let Spellings = [CXX11<"clang", "lto_visibility_public">];
  let Subjects = SubjectList<[Record]>;
  let Documentation = [LTOVisibilityDocs];
}

def AnyX86Interrupt : InheritableAttr, TargetSpecificAttr<TargetAnyX86> {
  // NOTE: If you add any additional spellings, ARMInterrupt's,
  // MSP430Interrupt's and MipsInterrupt's spellings must match.
  let Spellings = [GNU<"interrupt">];
  let Subjects = SubjectList<[HasFunctionProto]>;
  let ParseKind = "Interrupt";
  let HasCustomParsing = 1;
  let Documentation = [AnyX86InterruptDocs];
}

def X86ForceAlignArgPointer : InheritableAttr, TargetSpecificAttr<TargetX86> {
  let Spellings = [GNU<"force_align_arg_pointer">];
  // Technically, this appertains to a FunctionDecl, but the target-specific
  // code silently allows anything function-like (such as typedefs or function
  // pointers), but does not apply the attribute to them.
  let Documentation = [Undocumented];
}

def NoSanitize : InheritableAttr {
  let Spellings = [GNU<"no_sanitize">, CXX11<"clang", "no_sanitize">];
  let Args = [VariadicStringArgument<"Sanitizers">];
  let Subjects = SubjectList<[Function, ObjCMethod], ErrorDiag>;
  let Documentation = [NoSanitizeDocs];
  let AdditionalMembers = [{
    SanitizerMask getMask() const {
      SanitizerMask Mask = 0;
      for (auto SanitizerName : sanitizers()) {
        SanitizerMask ParsedMask =
            parseSanitizerValue(SanitizerName, /*AllowGroups=*/true);
        Mask |= expandSanitizerGroups(ParsedMask);
      }
      return Mask;
    }
  }];
}

// Attributes to disable a specific sanitizer. No new sanitizers should be added
// to this list; the no_sanitize attribute should be extended instead.
def NoSanitizeSpecific : InheritableAttr {
  let Spellings = [GCC<"no_address_safety_analysis">,
                   GCC<"no_sanitize_address">,
                   GCC<"no_sanitize_thread">,
                   GNU<"no_sanitize_memory">];
  let Subjects = SubjectList<[Function], ErrorDiag>;
  let Documentation = [NoSanitizeAddressDocs, NoSanitizeThreadDocs,
                       NoSanitizeMemoryDocs];
  let ASTNode = 0;
}

// C/C++ Thread safety attributes (e.g. for deadlock, data race checking)

def GuardedVar : InheritableAttr {
  let Spellings = [GNU<"guarded_var">];
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
  let Documentation = [Undocumented];
}

def PtGuardedVar : InheritableAttr {
  let Spellings = [GNU<"pt_guarded_var">];
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
  let Documentation = [Undocumented];
}

def Lockable : InheritableAttr {
  let Spellings = [GNU<"lockable">];
  let Subjects = SubjectList<[Record]>;
  let Documentation = [Undocumented];
  let ASTNode = 0;  // Replaced by Capability
}

def ScopedLockable : InheritableAttr {
  let Spellings = [GNU<"scoped_lockable">];
  let Subjects = SubjectList<[Record]>;
  let Documentation = [Undocumented];
}

def Capability : InheritableAttr {
  let Spellings = [GNU<"capability">, CXX11<"clang", "capability">,
                   GNU<"shared_capability">,
                   CXX11<"clang", "shared_capability">];
  let Subjects = SubjectList<[Record, TypedefName], ErrorDiag,
                             "ExpectedStructOrUnionOrTypedef">;
  let Args = [StringArgument<"Name">];
  let Accessors = [Accessor<"isShared",
                    [GNU<"shared_capability">,
                     CXX11<"clang","shared_capability">]>];
  let Documentation = [Undocumented];
  let AdditionalMembers = [{
    bool isMutex() const { return getName().equals_lower("mutex"); }
    bool isRole() const { return getName().equals_lower("role"); }
  }];
}

def AssertCapability : InheritableAttr {
  let Spellings = [GNU<"assert_capability">,
                   CXX11<"clang", "assert_capability">,
                   GNU<"assert_shared_capability">,
                   CXX11<"clang", "assert_shared_capability">];
  let Subjects = SubjectList<[Function]>;
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Args = [ExprArgument<"Expr">];
  let Accessors = [Accessor<"isShared",
                    [GNU<"assert_shared_capability">,
                     CXX11<"clang", "assert_shared_capability">]>];
  let Documentation = [AssertCapabilityDocs];
}

def AcquireCapability : InheritableAttr {
  let Spellings = [GNU<"acquire_capability">,
                   CXX11<"clang", "acquire_capability">,
                   GNU<"acquire_shared_capability">,
                   CXX11<"clang", "acquire_shared_capability">,
                   GNU<"exclusive_lock_function">,
                   GNU<"shared_lock_function">];
  let Subjects = SubjectList<[Function]>;
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Args = [VariadicExprArgument<"Args">];
  let Accessors = [Accessor<"isShared",
                    [GNU<"acquire_shared_capability">,
                     CXX11<"clang", "acquire_shared_capability">,
                     GNU<"shared_lock_function">]>];
  let Documentation = [AcquireCapabilityDocs];
}

def TryAcquireCapability : InheritableAttr {
  let Spellings = [GNU<"try_acquire_capability">,
                   CXX11<"clang", "try_acquire_capability">,
                   GNU<"try_acquire_shared_capability">,
                   CXX11<"clang", "try_acquire_shared_capability">];
  let Subjects = SubjectList<[Function],
                             ErrorDiag>;
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let Accessors = [Accessor<"isShared",
                    [GNU<"try_acquire_shared_capability">,
                     CXX11<"clang", "try_acquire_shared_capability">]>];
  let Documentation = [TryAcquireCapabilityDocs];
}

def ReleaseCapability : InheritableAttr {
  let Spellings = [GNU<"release_capability">,
                   CXX11<"clang", "release_capability">,
                   GNU<"release_shared_capability">,
                   CXX11<"clang", "release_shared_capability">,
                   GNU<"release_generic_capability">,
                   CXX11<"clang", "release_generic_capability">,
                   GNU<"unlock_function">];
  let Subjects = SubjectList<[Function]>;
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Args = [VariadicExprArgument<"Args">];
  let Accessors = [Accessor<"isShared",
                    [GNU<"release_shared_capability">,
                     CXX11<"clang", "release_shared_capability">]>,
                   Accessor<"isGeneric",
                     [GNU<"release_generic_capability">,
                      CXX11<"clang", "release_generic_capability">,
                      GNU<"unlock_function">]>];
  let Documentation = [ReleaseCapabilityDocs];
}

def RequiresCapability : InheritableAttr {
  let Spellings = [GNU<"requires_capability">,
                   CXX11<"clang", "requires_capability">,
                   GNU<"exclusive_locks_required">,
                   GNU<"requires_shared_capability">,
                   CXX11<"clang", "requires_shared_capability">,
                   GNU<"shared_locks_required">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function]>;
  let Accessors = [Accessor<"isShared", [GNU<"requires_shared_capability">,
                                         GNU<"shared_locks_required">,
                                CXX11<"clang","requires_shared_capability">]>];
  let Documentation = [Undocumented];
}

def NoThreadSafetyAnalysis : InheritableAttr {
  let Spellings = [GNU<"no_thread_safety_analysis">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def GuardedBy : InheritableAttr {
  let Spellings = [GNU<"guarded_by">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
  let Documentation = [Undocumented];
}

def PtGuardedBy : InheritableAttr {
  let Spellings = [GNU<"pt_guarded_by">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
  let Documentation = [Undocumented];
}

def AcquiredAfter : InheritableAttr {
  let Spellings = [GNU<"acquired_after">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
  let Documentation = [Undocumented];
}

def AcquiredBefore : InheritableAttr {
  let Spellings = [GNU<"acquired_before">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Field, SharedVar], WarnDiag,
                             "ExpectedFieldOrGlobalVar">;
  let Documentation = [Undocumented];
}

def AssertExclusiveLock : InheritableAttr {
  let Spellings = [GNU<"assert_exclusive_lock">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def AssertSharedLock : InheritableAttr {
  let Spellings = [GNU<"assert_shared_lock">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def ExclusiveTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"exclusive_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

// The first argument is an integer or boolean value specifying the return value
// of a successful lock acquisition.
def SharedTrylockFunction : InheritableAttr {
  let Spellings = [GNU<"shared_trylock_function">];
  let Args = [ExprArgument<"SuccessValue">, VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def LockReturned : InheritableAttr {
  let Spellings = [GNU<"lock_returned">];
  let Args = [ExprArgument<"Arg">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def LocksExcluded : InheritableAttr {
  let Spellings = [GNU<"locks_excluded">];
  let Args = [VariadicExprArgument<"Args">];
  let LateParsed = 1;
  let TemplateDependent = 1;
  let ParseArgumentsAsUnevaluated = 1;
  let DuplicatesAllowedWhileMerging = 1;
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

// C/C++ consumed attributes.

def Consumable : InheritableAttr {
  let Spellings = [GNU<"consumable">];
  let Subjects = SubjectList<[CXXRecord]>;
  let Args = [EnumArgument<"DefaultState", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
  let Documentation = [ConsumableDocs];
}

def ConsumableAutoCast : InheritableAttr {
  let Spellings = [GNU<"consumable_auto_cast_state">];
  let Subjects = SubjectList<[CXXRecord]>;
  let Documentation = [Undocumented];
}

def ConsumableSetOnRead : InheritableAttr {
  let Spellings = [GNU<"consumable_set_state_on_read">];
  let Subjects = SubjectList<[CXXRecord]>;
  let Documentation = [Undocumented];
}

def CallableWhen : InheritableAttr {
  let Spellings = [GNU<"callable_when">];
  let Subjects = SubjectList<[CXXMethod]>;
  let Args = [VariadicEnumArgument<"CallableStates", "ConsumedState",
                                   ["unknown", "consumed", "unconsumed"],
                                   ["Unknown", "Consumed", "Unconsumed"]>];
  let Documentation = [CallableWhenDocs];
}

def ParamTypestate : InheritableAttr {
  let Spellings = [GNU<"param_typestate">];
  let Subjects = SubjectList<[ParmVar]>;
  let Args = [EnumArgument<"ParamState", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
  let Documentation = [ParamTypestateDocs];
}

def ReturnTypestate : InheritableAttr {
  let Spellings = [GNU<"return_typestate">];
  let Subjects = SubjectList<[Function, ParmVar]>;
  let Args = [EnumArgument<"State", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
  let Documentation = [ReturnTypestateDocs];
}

def SetTypestate : InheritableAttr {
  let Spellings = [GNU<"set_typestate">];
  let Subjects = SubjectList<[CXXMethod]>;
  let Args = [EnumArgument<"NewState", "ConsumedState",
                           ["unknown", "consumed", "unconsumed"],
                           ["Unknown", "Consumed", "Unconsumed"]>];
  let Documentation = [SetTypestateDocs];
}

def TestTypestate : InheritableAttr {
  let Spellings = [GNU<"test_typestate">];
  let Subjects = SubjectList<[CXXMethod]>;
  let Args = [EnumArgument<"TestState", "ConsumedState",
                           ["consumed", "unconsumed"],
                           ["Consumed", "Unconsumed"]>];
  let Documentation = [TestTypestateDocs];
}

// Type safety attributes for `void *' pointers and type tags.

def ArgumentWithTypeTag : InheritableAttr {
  let Spellings = [GNU<"argument_with_type_tag">,
                   GNU<"pointer_with_type_tag">];
  let Args = [IdentifierArgument<"ArgumentKind">,
              UnsignedArgument<"ArgumentIdx">,
              UnsignedArgument<"TypeTagIdx">,
              BoolArgument<"IsPointer">];
  let HasCustomParsing = 1;
  let Documentation = [ArgumentWithTypeTagDocs, PointerWithTypeTagDocs];
}

def TypeTagForDatatype : InheritableAttr {
  let Spellings = [GNU<"type_tag_for_datatype">];
  let Args = [IdentifierArgument<"ArgumentKind">,
              TypeArgument<"MatchingCType">,
              BoolArgument<"LayoutCompatible">,
              BoolArgument<"MustBeNull">];
//  let Subjects = SubjectList<[Var], ErrorDiag>;
  let HasCustomParsing = 1;
  let Documentation = [TypeTagForDatatypeDocs];
}

// Microsoft-related attributes

def MSNoVTable : InheritableAttr, TargetSpecificAttr<TargetMicrosoftCXXABI> {
  let Spellings = [Declspec<"novtable">];
  let Subjects = SubjectList<[CXXRecord]>;
  let Documentation = [MSNoVTableDocs];
}

def : IgnoredAttr {
  let Spellings = [Declspec<"property">];
}

def MSStruct : InheritableAttr {
  let Spellings = [GCC<"ms_struct">];
  let Subjects = SubjectList<[Record]>;
  let Documentation = [Undocumented];
}

def DLLExport : InheritableAttr, TargetSpecificAttr<TargetWindows> {
  let Spellings = [Declspec<"dllexport">, GCC<"dllexport">];
  let Subjects = SubjectList<[Function, Var, CXXRecord, ObjCInterface]>;
  let Documentation = [DLLExportDocs];
}

def DLLImport : InheritableAttr, TargetSpecificAttr<TargetWindows> {
  let Spellings = [Declspec<"dllimport">, GCC<"dllimport">];
  let Subjects = SubjectList<[Function, Var, CXXRecord, ObjCInterface]>;
  let Documentation = [DLLImportDocs];
}

def SelectAny : InheritableAttr {
  let Spellings = [Declspec<"selectany">];
  let LangOpts = [MicrosoftExt];
  let Documentation = [Undocumented];
}

def Thread : Attr {
  let Spellings = [Declspec<"thread">];
  let LangOpts = [MicrosoftExt];
  let Documentation = [ThreadDocs];
  let Subjects = SubjectList<[Var]>;
}

def Win64 : IgnoredAttr {
  let Spellings = [Keyword<"__w64">];
  let LangOpts = [MicrosoftExt];
}

def Ptr32 : TypeAttr {
  let Spellings = [Keyword<"__ptr32">];
  let Documentation = [Undocumented];
}

def Ptr64 : TypeAttr {
  let Spellings = [Keyword<"__ptr64">];
  let Documentation = [Undocumented];
}

def SPtr : TypeAttr {
  let Spellings = [Keyword<"__sptr">];
  let Documentation = [Undocumented];
}

def UPtr : TypeAttr {
  let Spellings = [Keyword<"__uptr">];
  let Documentation = [Undocumented];
}

def MSInheritance : InheritableAttr {
  let LangOpts = [MicrosoftExt];
  let Args = [DefaultBoolArgument<"BestCase", 1>];
  let Spellings = [Keyword<"__single_inheritance">,
                   Keyword<"__multiple_inheritance">,
                   Keyword<"__virtual_inheritance">,
                   Keyword<"__unspecified_inheritance">];
  let AdditionalMembers = [{
  static bool hasVBPtrOffsetField(Spelling Inheritance) {
    return Inheritance == Keyword_unspecified_inheritance;
  }

  // Only member pointers to functions need a this adjustment, since it can be
  // combined with the field offset for data pointers.
  static bool hasNVOffsetField(bool IsMemberFunction, Spelling Inheritance) {
    return IsMemberFunction && Inheritance >= Keyword_multiple_inheritance;
  }

  static bool hasVBTableOffsetField(Spelling Inheritance) {
    return Inheritance >= Keyword_virtual_inheritance;
  }

  static bool hasOnlyOneField(bool IsMemberFunction,
                              Spelling Inheritance) {
    if (IsMemberFunction)
      return Inheritance <= Keyword_single_inheritance;
    return Inheritance <= Keyword_multiple_inheritance;
  }
  }];
  let Documentation = [MSInheritanceDocs];
}

def MSVtorDisp : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let Args = [UnsignedArgument<"vdm">];
  let SemaHandler = 0;

  let AdditionalMembers = [{
  enum Mode {
    Never,
    ForVBaseOverride,
    ForVFTable
  };

  Mode getVtorDispMode() const { return Mode(vdm); }
  }];
  let Documentation = [Undocumented];
}

def InitSeg : Attr {
  let Spellings = [Pragma<"", "init_seg">];
  let Args = [StringArgument<"Section">];
  let SemaHandler = 0;
  let Documentation = [InitSegDocs];
  let AdditionalMembers = [{
  void printPrettyPragma(raw_ostream &OS, const PrintingPolicy &Policy) const {
    OS << '(' << getSection() << ')';
  }
  }];
}

def LoopHint : Attr {
  /// #pragma clang loop <option> directive
  /// vectorize: vectorizes loop operations if State == Enable.
  /// vectorize_width: vectorize loop operations with width 'Value'.
  /// interleave: interleave multiple loop iterations if State == Enable.
  /// interleave_count: interleaves 'Value' loop interations.
  /// unroll: fully unroll loop if State == Enable.
  /// unroll_count: unrolls loop 'Value' times.
  /// distribute: attempt to distribute loop if State == Enable

  /// #pragma unroll <argument> directive
  /// <no arg>: fully unrolls loop.
  /// boolean: fully unrolls loop if State == Enable.
  /// expression: unrolls loop 'Value' times.

  let Spellings = [Pragma<"clang", "loop">, Pragma<"", "unroll">,
                   Pragma<"", "nounroll">];

  /// State of the loop optimization specified by the spelling.
  let Args = [EnumArgument<"Option", "OptionType",
                          ["vectorize", "vectorize_width", "interleave", "interleave_count",
                           "unroll", "unroll_count", "distribute"],
                          ["Vectorize", "VectorizeWidth", "Interleave", "InterleaveCount",
                           "Unroll", "UnrollCount", "Distribute"]>,
              EnumArgument<"State", "LoopHintState",
                           ["enable", "disable", "numeric", "assume_safety", "full"],
                           ["Enable", "Disable", "Numeric", "AssumeSafety", "Full"]>,
              ExprArgument<"Value">];

  let AdditionalMembers = [{
  static const char *getOptionName(int Option) {
    switch(Option) {
    case Vectorize: return "vectorize";
    case VectorizeWidth: return "vectorize_width";
    case Interleave: return "interleave";
    case InterleaveCount: return "interleave_count";
    case Unroll: return "unroll";
    case UnrollCount: return "unroll_count";
    case Distribute: return "distribute";
    }
    llvm_unreachable("Unhandled LoopHint option.");
  }

  void printPrettyPragma(raw_ostream &OS, const PrintingPolicy &Policy) const {
    unsigned SpellingIndex = getSpellingListIndex();
    // For "#pragma unroll" and "#pragma nounroll" the string "unroll" or
    // "nounroll" is already emitted as the pragma name.
    if (SpellingIndex == Pragma_nounroll)
      return;
    else if (SpellingIndex == Pragma_unroll) {
      OS << getValueString(Policy);
      return;
    }

    assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
    OS << getOptionName(option) << getValueString(Policy);
  }

  // Return a string containing the loop hint argument including the
  // enclosing parentheses.
  std::string getValueString(const PrintingPolicy &Policy) const {
    std::string ValueName;
    llvm::raw_string_ostream OS(ValueName);
    OS << "(";
    if (state == Numeric)
      value->printPretty(OS, nullptr, Policy);
    else if (state == Enable)
      OS << "enable";
    else if (state == Full)
      OS << "full";
    else if (state == AssumeSafety)
      OS << "assume_safety";
    else
      OS << "disable";
    OS << ")";
    return OS.str();
  }

  // Return a string suitable for identifying this attribute in diagnostics.
  std::string getDiagnosticName(const PrintingPolicy &Policy) const {
    unsigned SpellingIndex = getSpellingListIndex();
    if (SpellingIndex == Pragma_nounroll)
      return "#pragma nounroll";
    else if (SpellingIndex == Pragma_unroll)
      return "#pragma unroll" + (option == UnrollCount ? getValueString(Policy) : "");

    assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
    return getOptionName(option) + getValueString(Policy);
  }
  }];

  let Documentation = [LoopHintDocs, UnrollHintDocs];
}

def CapturedRecord : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def OMPThreadPrivateDecl : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def OMPCaptureNoInit : InheritableAttr {
  // This attribute has no spellings as it is only ever created implicitly.
  let Spellings = [];
  let SemaHandler = 0;
  let Documentation = [Undocumented];
}

def OMPDeclareSimdDecl : Attr {
  let Spellings = [Pragma<"omp", "declare simd">];
  let Subjects = SubjectList<[Function]>;
  let SemaHandler = 0;
  let HasCustomParsing = 1;
  let Documentation = [OMPDeclareSimdDocs];
  let Args = [
    EnumArgument<"BranchState", "BranchStateTy",
                 [ "", "inbranch", "notinbranch" ],
                 [ "BS_Undefined", "BS_Inbranch", "BS_Notinbranch" ]>,
    ExprArgument<"Simdlen">, VariadicExprArgument<"Uniforms">,
    VariadicExprArgument<"Aligneds">, VariadicExprArgument<"Alignments">,
    VariadicExprArgument<"Linears">, VariadicUnsignedArgument<"Modifiers">,
    VariadicExprArgument<"Steps">
  ];
  let AdditionalMembers = [{
    void printPrettyPragma(raw_ostream & OS, const PrintingPolicy &Policy)
        const {
      if (getBranchState() != BS_Undefined)
        OS << ConvertBranchStateTyToStr(getBranchState()) << " ";
      if (auto *E = getSimdlen()) {
        OS << "simdlen(";
        E->printPretty(OS, nullptr, Policy);
        OS << ") ";
      }
      if (uniforms_size() > 0) {
        OS << "uniform";
        StringRef Sep = "(";
        for (auto *E : uniforms()) {
          OS << Sep;
          E->printPretty(OS, nullptr, Policy);
          Sep = ", ";
        }
        OS << ") ";
      }
      alignments_iterator NI = alignments_begin();
      for (auto *E : aligneds()) {
        OS << "aligned(";
        E->printPretty(OS, nullptr, Policy);
        if (*NI) {
          OS << ": ";
          (*NI)->printPretty(OS, nullptr, Policy);
        }
        OS << ") ";
        ++NI;
      }
      steps_iterator I = steps_begin();
      modifiers_iterator MI = modifiers_begin();
      for (auto *E : linears()) {
        OS << "linear(";
        if (*MI != OMPC_LINEAR_unknown)
          OS << getOpenMPSimpleClauseTypeName(OMPC_linear, *MI) << "(";
        E->printPretty(OS, nullptr, Policy);
        if (*MI != OMPC_LINEAR_unknown)
          OS << ")";
        if (*I) {
          OS << ": ";
          (*I)->printPretty(OS, nullptr, Policy);
        }
        OS << ") ";
        ++I;
        ++MI;
      }
    }
  }];
}

def OMPDeclareTargetDecl : Attr {
  let Spellings = [Pragma<"omp", "declare target">];
  let SemaHandler = 0;
  let Documentation = [OMPDeclareTargetDocs];
  let Args = [
    EnumArgument<"MapType", "MapTypeTy",
                 [ "to", "link" ],
                 [ "MT_To", "MT_Link" ]>
  ];
  let AdditionalMembers = [{
    void printPrettyPragma(raw_ostream &OS, const PrintingPolicy &Policy) const {
      // Use fake syntax because it is for testing and debugging purpose only.
      if (getMapType() != MT_To)
        OS << ConvertMapTypeTyToStr(getMapType()) << " ";
    }
  }];
}

def InternalLinkage : InheritableAttr {
  let Spellings = [GNU<"internal_linkage">, CXX11<"clang", "internal_linkage">];
  let Subjects = SubjectList<[Var, Function, CXXRecord]>;
  let Documentation = [InternalLinkageDocs];
}
