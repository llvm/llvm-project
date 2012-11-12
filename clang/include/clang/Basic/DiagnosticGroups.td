//==--- DiagnosticGroups.td - Diagnostic Group Definitions ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

def ImplicitFunctionDeclare : DiagGroup<"implicit-function-declaration">;
def ImplicitInt : DiagGroup<"implicit-int">;

// Aggregation warning settings.
def Implicit : DiagGroup<"implicit", [
    ImplicitFunctionDeclare,
    ImplicitInt
]>;

// Empty DiagGroups are recognized by clang but ignored.
def : DiagGroup<"abi">;
def : DiagGroup<"address">;
def AddressOfTemporary : DiagGroup<"address-of-temporary">;
def : DiagGroup<"aggregate-return">;
def AmbigMemberTemplate : DiagGroup<"ambiguous-member-template">;
def Availability : DiagGroup<"availability">;
def Section : DiagGroup<"section">;
def AutoImport : DiagGroup<"auto-import">;
def ConstantConversion : DiagGroup<"constant-conversion">;
def LiteralConversion : DiagGroup<"literal-conversion">;
def StringConversion : DiagGroup<"string-conversion">;
def SignConversion : DiagGroup<"sign-conversion">;
def BoolConversion : DiagGroup<"bool-conversion">;
def IntConversion : DiagGroup<"int-conversion">;
def EnumConversion : DiagGroup<"enum-conversion">;
def NonLiteralNullConversion : DiagGroup<"non-literal-null-conversion">;
def NullConversion : DiagGroup<"null-conversion">;
def ImplicitConversionFloatingPointToBool :
  DiagGroup<"implicit-conversion-floating-point-to-bool">;
def BuiltinRequiresHeader : DiagGroup<"builtin-requires-header">;
def CXXCompat: DiagGroup<"c++-compat">;
def CastAlign : DiagGroup<"cast-align">;
def : DiagGroup<"cast-qual">;
def : DiagGroup<"char-align">;
def Comment : DiagGroup<"comment">;
def : DiagGroup<"ctor-dtor-privacy">;
def DefaultArgSpecialMember : DiagGroup<"default-arg-special-member">;
def GNUDesignator : DiagGroup<"gnu-designator">;

def DeleteNonVirtualDtor : DiagGroup<"delete-non-virtual-dtor">;
def AbstractFinalClass : DiagGroup<"abstract-final-class">;

def DeprecatedDeclarations : DiagGroup<"deprecated-declarations">;
def DeprecatedWritableStr : DiagGroup<"deprecated-writable-strings">;
def Deprecated : DiagGroup<"deprecated", [ DeprecatedDeclarations] >,
                 DiagCategory<"Deprecations">;

def DeprecatedImplementations :DiagGroup<"deprecated-implementations">;

def : DiagGroup<"disabled-optimization">;
def : DiagGroup<"discard-qual">;
def : DiagGroup<"div-by-zero">;

def DocumentationHTML : DiagGroup<"documentation-html">;
def DocumentationPedantic : DiagGroup<"documentation-pedantic">;
def DocumentationDeprecatedSync : DiagGroup<"documentation-deprecated-sync">;
def Documentation : DiagGroup<"documentation",
                              [DocumentationHTML,
                               DocumentationDeprecatedSync]>;

def EmptyBody : DiagGroup<"empty-body">;
def ExtraTokens : DiagGroup<"extra-tokens">;
def CXX11ExtraSemi : DiagGroup<"c++11-extra-semi">;
def ExtraSemi : DiagGroup<"extra-semi", [CXX11ExtraSemi]>;

def FormatExtraArgs : DiagGroup<"format-extra-args">;
def FormatZeroLength : DiagGroup<"format-zero-length">;

def CXX98CompatBindToTemporaryCopy :
  DiagGroup<"c++98-compat-bind-to-temporary-copy">;
def CXX98CompatLocalTypeTemplateArgs :
  DiagGroup<"c++98-compat-local-type-template-args">;
def CXX98CompatUnnamedTypeTemplateArgs :
  DiagGroup<"c++98-compat-unnamed-type-template-args">;

def CXX98Compat : DiagGroup<"c++98-compat",
                            [CXX98CompatBindToTemporaryCopy,
                             CXX98CompatLocalTypeTemplateArgs,
                             CXX98CompatUnnamedTypeTemplateArgs]>;
// Warnings for C++11 features which are Extensions in C++98 mode.
def CXX98CompatPedantic : DiagGroup<"c++98-compat-pedantic", [CXX98Compat]>;

def CXX11Narrowing : DiagGroup<"c++11-narrowing">;

// Original name of this warning in Clang
def : DiagGroup<"c++0x-narrowing", [CXX11Narrowing]>;

// Name of this warning in GCC
def : DiagGroup<"narrowing", [CXX11Narrowing]>;

def CXX11CompatReservedUserDefinedLiteral :
  DiagGroup<"c++11-compat-reserved-user-defined-literal">;
def ReservedUserDefinedLiteral :
  DiagGroup<"reserved-user-defined-literal",
            [CXX11CompatReservedUserDefinedLiteral]>;

def CXX11Compat : DiagGroup<"c++11-compat",
                            [CXX11Narrowing,
                             CXX11CompatReservedUserDefinedLiteral]>;
def : DiagGroup<"c++0x-compat", [CXX11Compat]>;

def : DiagGroup<"effc++">;
def DivZero : DiagGroup<"division-by-zero">;
def ExitTimeDestructors : DiagGroup<"exit-time-destructors">;
def FlexibleArrayExtensions : DiagGroup<"flexible-array-extensions">;
def FourByteMultiChar : DiagGroup<"four-char-constants">;
def GlobalConstructors : DiagGroup<"global-constructors">;
def BitwiseOpParentheses: DiagGroup<"bitwise-op-parentheses">;
def LogicalOpParentheses: DiagGroup<"logical-op-parentheses">;
def ShiftOpParentheses: DiagGroup<"shift-op-parentheses">;
def DanglingElse: DiagGroup<"dangling-else">;
def IgnoredQualifiers : DiagGroup<"ignored-qualifiers">;
def : DiagGroup<"import">;
def IncompatiblePointerTypes : DiagGroup<"incompatible-pointer-types">;
def IncompleteUmbrella : DiagGroup<"incomplete-umbrella">;
def KNRPromotedParameter : DiagGroup<"knr-promoted-parameter">;
def : DiagGroup<"init-self">;
def : DiagGroup<"inline">;
def : DiagGroup<"invalid-pch">;
def LiteralRange : DiagGroup<"literal-range">;
def LocalTypeTemplateArgs : DiagGroup<"local-type-template-args",
                                      [CXX98CompatLocalTypeTemplateArgs]>;
def MalformedWarningCheck : DiagGroup<"malformed-warning-check">;
def Main : DiagGroup<"main">;
def MainReturnType : DiagGroup<"main-return-type">;
def MissingBraces : DiagGroup<"missing-braces">;
def MissingDeclarations: DiagGroup<"missing-declarations">;
def : DiagGroup<"missing-format-attribute">;
def : DiagGroup<"missing-include-dirs">;
def MultiChar : DiagGroup<"multichar">;
def : DiagGroup<"nested-externs">;
def CXX11LongLong : DiagGroup<"c++11-long-long">;
def LongLong : DiagGroup<"long-long", [CXX11LongLong]>;
def MismatchedParameterTypes : DiagGroup<"mismatched-parameter-types">;
def MismatchedReturnTypes : DiagGroup<"mismatched-return-types">;
def MismatchedTags : DiagGroup<"mismatched-tags">;
def MissingFieldInitializers : DiagGroup<"missing-field-initializers">;
def ModuleBuild : DiagGroup<"module-build">;
def NullCharacter : DiagGroup<"null-character">;
def NullDereference : DiagGroup<"null-dereference">;
def InitializerOverrides : DiagGroup<"initializer-overrides">;
def NonNull : DiagGroup<"nonnull">;
def : DiagGroup<"nonportable-cfstrings">;
def NonVirtualDtor : DiagGroup<"non-virtual-dtor">;
def OveralignedType : DiagGroup<"over-aligned">;
def : DiagGroup<"old-style-cast">;
def : DiagGroup<"old-style-definition">;
def OutOfLineDeclaration : DiagGroup<"out-of-line-declaration">;
def : DiagGroup<"overflow">;
def ForwardClassReceiver : DiagGroup<"receiver-forward-class">;
def MethodAccess : DiagGroup<"objc-method-access">;
def ObjCReceiver : DiagGroup<"receiver-expr">;
def OverlengthStrings : DiagGroup<"overlength-strings">;
def OverloadedVirtual : DiagGroup<"overloaded-virtual">;
def PrivateExtern : DiagGroup<"private-extern">;
def SelTypeCast : DiagGroup<"cast-of-sel-type">;
def BadFunctionCast : DiagGroup<"bad-function-cast">;
def ObjCPropertyImpl : DiagGroup<"objc-property-implementation">;
def ObjCPropertyNoAttribute : DiagGroup<"objc-property-no-attribute">;
def ObjCMissingSuperCalls : DiagGroup<"objc-missing-super-calls">;
def ObjCRetainBlockProperty : DiagGroup<"objc-noncopy-retain-block-property">;
def ObjCReadonlyPropertyHasSetter : DiagGroup<"objc-readonly-with-setter-property">;
def ObjCRootClass : DiagGroup<"objc-root-class">;
def Packed : DiagGroup<"packed">;
def Padded : DiagGroup<"padded">;
def PointerArith : DiagGroup<"pointer-arith">;
def PoundWarning : DiagGroup<"#warnings">,
                   DiagCategory<"#warning Directive">;
def PoundPragmaMessage : DiagGroup<"#pragma-messages">,
                         DiagCategory<"#pragma message Directive">;
def : DiagGroup<"pointer-to-int-cast">;
def : DiagGroup<"redundant-decls">;
def ReturnTypeCLinkage : DiagGroup<"return-type-c-linkage">;
def ReturnType : DiagGroup<"return-type", [ReturnTypeCLinkage]>;
def BindToTemporaryCopy : DiagGroup<"bind-to-temporary-copy",
                                    [CXX98CompatBindToTemporaryCopy]>;
def SelfAssignmentField : DiagGroup<"self-assign-field">;
def SelfAssignment : DiagGroup<"self-assign", [SelfAssignmentField]>;
def SemiBeforeMethodBody : DiagGroup<"semicolon-before-method-body">;
def Sentinel : DiagGroup<"sentinel">;
def MissingMethodReturnType : DiagGroup<"missing-method-return-type">;
def : DiagGroup<"sequence-point">;
def Shadow : DiagGroup<"shadow">;
def Shorten64To32 : DiagGroup<"shorten-64-to-32">;
def : DiagGroup<"sign-promo">;
def SignCompare : DiagGroup<"sign-compare">;
def : DiagGroup<"stack-protector">;
def : DiagGroup<"switch-default">;
def : DiagGroup<"synth">;
def SizeofArrayArgument : DiagGroup<"sizeof-array-argument">;
def StringPlusInt : DiagGroup<"string-plus-int">;
def StrncatSize : DiagGroup<"strncat-size">;
def TautologicalOutOfRangeCompare : DiagGroup<"tautological-constant-out-of-range-compare">;
def TautologicalCompare : DiagGroup<"tautological-compare",
                                    [TautologicalOutOfRangeCompare]>;
def HeaderHygiene : DiagGroup<"header-hygiene">;
def DuplicateDeclSpecifier : DiagGroup<"duplicate-decl-specifier">;
def CompareDistinctPointerType : DiagGroup<"compare-distinct-pointer-types">;

// Preprocessor warnings.
def AmbiguousMacro : DiagGroup<"ambiguous-macro">;

// Just silence warnings about -Wstrict-aliasing for now.
def : DiagGroup<"strict-aliasing=0">;
def : DiagGroup<"strict-aliasing=1">;
def : DiagGroup<"strict-aliasing=2">;
def : DiagGroup<"strict-aliasing">;

// Just silence warnings about -Wstrict-overflow for now.
def : DiagGroup<"strict-overflow=0">;
def : DiagGroup<"strict-overflow=1">;
def : DiagGroup<"strict-overflow=2">;
def : DiagGroup<"strict-overflow=3">;
def : DiagGroup<"strict-overflow=4">;
def : DiagGroup<"strict-overflow=5">;
def : DiagGroup<"strict-overflow">;

def InvalidOffsetof : DiagGroup<"invalid-offsetof">;
def LambdaExtensions : DiagGroup<"lambda-extensions">;
def : DiagGroup<"strict-prototypes">;
def StrictSelector : DiagGroup<"strict-selector-match">;
def MethodDuplicate : DiagGroup<"duplicate-method-match">;
def CoveredSwitchDefault : DiagGroup<"covered-switch-default">;
def SwitchEnum     : DiagGroup<"switch-enum">;
def Switch         : DiagGroup<"switch">;
def ImplicitFallthroughPerFunction :
  DiagGroup<"implicit-fallthrough-per-function">;
def ImplicitFallthrough  : DiagGroup<"implicit-fallthrough",
                                     [ImplicitFallthroughPerFunction]>;
def InvalidPPToken : DiagGroup<"invalid-pp-token">;
def Trigraphs      : DiagGroup<"trigraphs">;

def : DiagGroup<"type-limits">;
def Unicode  : DiagGroup<"unicode">;
def UninitializedMaybe : DiagGroup<"conditional-uninitialized">;
def UninitializedSometimes : DiagGroup<"sometimes-uninitialized">;
def Uninitialized  : DiagGroup<"uninitialized", [UninitializedSometimes]>;
def UnknownPragmas : DiagGroup<"unknown-pragmas">;
def NSobjectAttribute : DiagGroup<"NSObject-attribute">;
def UnknownAttributes : DiagGroup<"attributes">;
def IgnoredAttributes : DiagGroup<"ignored-attributes">;
def UnnamedTypeTemplateArgs : DiagGroup<"unnamed-type-template-args",
                                        [CXX98CompatUnnamedTypeTemplateArgs]>;
def UnusedArgument : DiagGroup<"unused-argument">;
def UnusedComparison : DiagGroup<"unused-comparison">;
def UnusedExceptionParameter : DiagGroup<"unused-exception-parameter">;
def UnneededInternalDecl : DiagGroup<"unneeded-internal-declaration">;
def UnneededMemberFunction : DiagGroup<"unneeded-member-function">;
def UnusedPrivateField : DiagGroup<"unused-private-field">;
def UnusedFunction : DiagGroup<"unused-function", [UnneededInternalDecl]>;
def UnusedMemberFunction : DiagGroup<"unused-member-function",
                                     [UnneededMemberFunction]>;
def UnusedLabel : DiagGroup<"unused-label">;
def UnusedParameter : DiagGroup<"unused-parameter">;
def UnusedResult : DiagGroup<"unused-result">;
def UnusedValue : DiagGroup<"unused-value", [UnusedComparison, UnusedResult]>;
def UnusedVariable : DiagGroup<"unused-variable">;
def UsedButMarkedUnused : DiagGroup<"used-but-marked-unused">;
def UserDefinedLiterals : DiagGroup<"user-defined-literals">;
def ReadOnlySetterAttrs : DiagGroup<"readonly-setter-attrs">;
def Reorder : DiagGroup<"reorder">;
def UndeclaredSelector : DiagGroup<"undeclared-selector">;
def ImplicitAtomic : DiagGroup<"implicit-atomic-properties">;
def CustomAtomic : DiagGroup<"custom-atomic-properties">;
def AtomicProperties : DiagGroup<"atomic-properties",
                                 [ImplicitAtomic, CustomAtomic]>;
// FIXME: Remove arc-abi once an Xcode is released that doesn't pass this flag.
def : DiagGroup<"arc-abi">;
def ARCUnsafeRetainedAssign : DiagGroup<"arc-unsafe-retained-assign">;
def ARCRetainCycles : DiagGroup<"arc-retain-cycles">;
def ARCNonPodMemAccess : DiagGroup<"arc-non-pod-memaccess">;
def AutomaticReferenceCounting : DiagGroup<"arc",
                                           [ARCUnsafeRetainedAssign,
                                            ARCRetainCycles,
                                            ARCNonPodMemAccess]>;
def ARCRepeatedUseOfWeakMaybe : DiagGroup<"arc-maybe-repeated-use-of-weak">;
def ARCRepeatedUseOfWeak : DiagGroup<"arc-repeated-use-of-weak",
                                     [ARCRepeatedUseOfWeakMaybe]>;
def Selector : DiagGroup<"selector">;
def Protocol : DiagGroup<"protocol">;
def SuperSubClassMismatch : DiagGroup<"super-class-method-mismatch">;
def OverridingMethodMismatch : DiagGroup<"overriding-method-mismatch">;
def VariadicMacros : DiagGroup<"variadic-macros">;
def VectorConversion : DiagGroup<"vector-conversion">;      // clang specific
def VexingParse : DiagGroup<"vexing-parse">;
def VLA : DiagGroup<"vla">;
def VolatileRegisterVar : DiagGroup<"volatile-register-var">;
def Visibility : DiagGroup<"visibility">;
def ZeroLengthArray : DiagGroup<"zero-length-array">;

// GCC calls -Wdeprecated-writable-strings -Wwrite-strings.
def GCCWriteStrings : DiagGroup<"write-strings" , [DeprecatedWritableStr]>;

def CharSubscript : DiagGroup<"char-subscripts">;
def LargeByValueCopy : DiagGroup<"large-by-value-copy">;
def DuplicateArgDecl : DiagGroup<"duplicate-method-arg">;

// Aggregation warning settings.

// -Widiomatic-parentheses contains warnings about 'idiomatic'
// missing parentheses;  it is off by default.  We do not include it
// in -Wparentheses because most users who use -Wparentheses explicitly
// do not want these warnings.
def ParenthesesOnEquality : DiagGroup<"parentheses-equality">;
def Parentheses : DiagGroup<"parentheses",
                            [LogicalOpParentheses,
                             BitwiseOpParentheses,
                             ShiftOpParentheses,
                             ParenthesesOnEquality,
                             DanglingElse]>;

// -Wconversion has its own warnings, but we split a few out for
// legacy reasons:
//   - some people want just 64-to-32 warnings
//   - conversion warnings with constant sources are on by default
//   - conversion warnings for literals are on by default
//   - bool-to-pointer conversion warnings are on by default
//   - __null-to-integer conversion warnings are on by default
def Conversion : DiagGroup<"conversion",
                           [BoolConversion,
                            ConstantConversion,
                            EnumConversion,
                            Shorten64To32,
                            IntConversion,
                            LiteralConversion,
                            NonLiteralNullConversion, // (1-1)->pointer (etc)
                            NullConversion, // NULL->non-pointer
                            SignConversion,
                            StringConversion]>,
                 DiagCategory<"Value Conversion Issue">;

def Unused : DiagGroup<"unused",
                       [UnusedArgument, UnusedFunction, UnusedLabel,
                        // UnusedParameter, (matches GCC's behavior)
                        // UnusedMemberFunction, (clean-up llvm before enabling)
                        UnusedPrivateField,
                        UnusedValue, UnusedVariable]>,
                        DiagCategory<"Unused Entity Issue">;

// Format settings.
def FormatInvalidSpecifier : DiagGroup<"format-invalid-specifier">;
def FormatSecurity : DiagGroup<"format-security">;
def FormatNonStandard : DiagGroup<"format-non-iso">;
def FormatY2K : DiagGroup<"format-y2k">;
def Format : DiagGroup<"format",
                       [FormatExtraArgs, FormatZeroLength, NonNull,
                        FormatSecurity, FormatY2K, FormatInvalidSpecifier]>,
             DiagCategory<"Format String Issue">;
def FormatNonLiteral : DiagGroup<"format-nonliteral", [FormatSecurity]>;
def Format2 : DiagGroup<"format=2",
                        [FormatNonLiteral, FormatSecurity, FormatY2K]>;

def TypeSafety : DiagGroup<"type-safety">;

def IntToPointerCast : DiagGroup<"int-to-pointer-cast">;

def Extra : DiagGroup<"extra", [
    MissingFieldInitializers,
    IgnoredQualifiers,
    InitializerOverrides,
    SemiBeforeMethodBody,
    MissingMethodReturnType,
    SignCompare,
    UnusedParameter
  ]>;

def Most : DiagGroup<"most", [
    CharSubscript,
    Comment,
    DeleteNonVirtualDtor,
    Format,
    Implicit,
    IntToPointerCast,
    MismatchedTags,
    MissingBraces,
    MultiChar,
    Reorder,
    ReturnType,
    SelfAssignment,
    SizeofArrayArgument,
    StringPlusInt,
    Trigraphs,
    Uninitialized,
    UnknownPragmas,
    Unused,
    VolatileRegisterVar,
    ObjCMissingSuperCalls,
    OverloadedVirtual,
    PrivateExtern,
    SelTypeCast
 ]>;

// Thread Safety warnings 
def ThreadSafetyAttributes : DiagGroup<"thread-safety-attributes">;
def ThreadSafetyAnalysis   : DiagGroup<"thread-safety-analysis">;
def ThreadSafetyPrecise    : DiagGroup<"thread-safety-precise">;
def ThreadSafety : DiagGroup<"thread-safety",
                             [ThreadSafetyAttributes, 
                              ThreadSafetyAnalysis,
                              ThreadSafetyPrecise]>;

// Note that putting warnings in -Wall will not disable them by default. If a
// warning should be active _only_ when -Wall is passed in, mark it as
// DefaultIgnore in addition to putting it here.
def : DiagGroup<"all", [Most, Parentheses, Switch]>;

// Warnings enabled by -pedantic.  This is magically filled in by TableGen.
def Pedantic : DiagGroup<"pedantic">;

// Aliases.
def : DiagGroup<"", [Extra]>;                   // -W = -Wextra
def : DiagGroup<"endif-labels", [ExtraTokens]>; // -Wendif-labels=-Wendif-tokens
def : DiagGroup<"comments", [Comment]>;         // -Wcomments = -Wcomment
def : DiagGroup<"conversion-null",
                [NullConversion]>; // -Wconversion-null = -Wnull-conversion
def : DiagGroup<"bool-conversions",
                [BoolConversion]>; // -Wbool-conversions  = -Wbool-conversion
def : DiagGroup<"int-conversions",
                [IntConversion]>; // -Wint-conversions = -Wint-conversion
def : DiagGroup<"vector-conversions",
                [VectorConversion]>; // -Wvector-conversions = -Wvector-conversion

// A warning group for warnings that we want to have on by default in clang,
// but which aren't on by default in GCC.
def NonGCC : DiagGroup<"non-gcc",
    [SignCompare, Conversion, LiteralRange]>;

// A warning group for warnings about using C++11 features as extensions in
// earlier C++ versions.
def CXX11 : DiagGroup<"c++11-extensions", [CXX11ExtraSemi, CXX11LongLong]>;

def : DiagGroup<"c++0x-extensions", [CXX11]>;
def DelegatingCtorCycles :
  DiagGroup<"delegating-ctor-cycles">;

// A warning group for warnings about using C11 features as extensions.
def C11 : DiagGroup<"c11-extensions">;

// A warning group for warnings about using C99 features as extensions.
def C99 : DiagGroup<"c99-extensions">;

// A warning group for warnings about GCC extensions.
def GNU : DiagGroup<"gnu", [GNUDesignator, VLA, ZeroLengthArray]>;
// A warning group for warnings about code that clang accepts but gcc doesn't.
def GccCompat : DiagGroup<"gcc-compat">;

// A warning group for warnings about Microsoft extensions.
def Microsoft : DiagGroup<"microsoft">;

def ObjCNonUnifiedException : DiagGroup<"objc-nonunified-exceptions">;

def ObjCProtocolMethodImpl : DiagGroup<"objc-protocol-method-implementation">;

// ObjC API warning groups.
def ObjCRedundantLiteralUse : DiagGroup<"objc-redundant-literal-use">;
def ObjCRedundantAPIUse : DiagGroup<"objc-redundant-api-use", [
    ObjCRedundantLiteralUse
  ]>;

def ObjCCocoaAPI : DiagGroup<"objc-cocoa-api", [
    ObjCRedundantAPIUse
  ]>;

def ObjCStringComparison : DiagGroup<"objc-string-compare">;
def ObjCLiteralComparison : DiagGroup<"objc-literal-compare", [
    ObjCStringComparison
  ]>;

// Inline ASM warnings.
def ASMOperandWidths : DiagGroup<"asm-operand-widths">;
def ASM : DiagGroup<"asm", [
    ASMOperandWidths
  ]>;
