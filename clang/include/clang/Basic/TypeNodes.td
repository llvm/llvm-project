class Type<bit abstract = 0> {
  bit Abstract = abstract;
} 

class DerivedType<Type base, bit abstract = 0> : Type<abstract> {
	Type Base = base;
}

/// A type node that is only used to represent dependent types in C++.  For
/// example, DependentTemplateSpecializationType is used to represent types
/// where the base template-id is dependent (such as `T::foo<U>`).  Code
/// that only works with non-dependent types can ignore these type nodes.
class AlwaysDependent {}

/// A type node that is never used to represent a canonical type, which is to
/// say that it always represents some sort of type "sugar" which can
/// (supposedly) be erased without affecting the formal behavior of the
/// language.  For example, in standard C/C++, typedefs do not introduce new
/// types and do not affect the semantics of the program.  Code that only
/// works with canonical types can ignore these type nodes.
///
/// Note that this simple story about non-canonical types is not the whole
/// truth.  Languages and extensions often have formation rules which differ
/// based on how a type is spelled and which therefore are not consistent
/// with immediately stipping away type sugar.  More critically, attributes on
/// typedefs can have semantic impacts in ways that are only reflected in our
/// AST by preserving the typedef sugar; for example, we do not otherwise
/// represent the alignment attribute on typedefs, and so it is necessary to
/// preserve typedef structure into most parts of IR generation.
class NeverCanonical {}

/// A type node that only represents a canonical type in some dependent cases.
/// For example, `std::vector<int>` (a TemplateSpecializationType) is
/// considered to be a non-canonical representation for the RecordType
/// referencing the concrete ClassTemplateSpecializationDecl; but
/// `std::vector<T>` cannot be resolved to a concrete specialization
/// and so remains canonical.  Code which only works with non-dependent
/// canonical types can ignore these nodes.
class NeverCanonicalUnlessDependent {}

/// A type node which never has component type structure.  Some code may be
/// able to operate on leaf types faster than they can on non-leaf types.
///
/// For example, the function type `void (int)` is not a leaf type because it
/// is structurally composed of component types (`void` and `int`).
///
/// A struct type is a leaf type because its field types are not part of its
/// type-expression.
///
/// Nodes like `TypedefType` which are syntactically leaves but can desugar
/// to types that may not be leaves should not declare this.
class LeafType {}

def BuiltinType : Type, LeafType;
def ComplexType : Type;
def PointerType : Type;
def BlockPointerType : Type;
def ReferenceType : Type<1>;
def LValueReferenceType : DerivedType<ReferenceType>;
def RValueReferenceType : DerivedType<ReferenceType>;
def MemberPointerType : Type;
def ArrayType : Type<1>;
def ConstantArrayType : DerivedType<ArrayType>;
def IncompleteArrayType : DerivedType<ArrayType>;
def VariableArrayType : DerivedType<ArrayType>;
def DependentSizedArrayType : DerivedType<ArrayType>, AlwaysDependent;
def DependentSizedExtVectorType : Type, AlwaysDependent;
def DependentAddressSpaceType : Type, AlwaysDependent;
def VectorType : Type;
def DependentVectorType : Type, AlwaysDependent;
def ExtVectorType : DerivedType<VectorType>;
def FunctionType : Type<1>;
def FunctionProtoType : DerivedType<FunctionType>;
def FunctionNoProtoType : DerivedType<FunctionType>;
def UnresolvedUsingType : Type, AlwaysDependent;
def ParenType : Type, NeverCanonical;
def TypedefType : Type, NeverCanonical;
def MacroQualifiedType : Type, NeverCanonical;
def AdjustedType : Type, NeverCanonical;
def DecayedType : DerivedType<AdjustedType>, NeverCanonical;
def TypeOfExprType : Type, NeverCanonicalUnlessDependent;
def TypeOfType : Type, NeverCanonicalUnlessDependent;
def DecltypeType : Type, NeverCanonicalUnlessDependent;
def UnaryTransformType : Type, NeverCanonicalUnlessDependent;
def TagType : Type<1>;
def RecordType : DerivedType<TagType>, LeafType;
def EnumType : DerivedType<TagType>, LeafType;
def ElaboratedType : Type, NeverCanonical;
def AttributedType : Type, NeverCanonical;
def TemplateTypeParmType : Type, AlwaysDependent, LeafType;
def SubstTemplateTypeParmType : Type, NeverCanonical;
def SubstTemplateTypeParmPackType : Type, AlwaysDependent;
def TemplateSpecializationType : Type, NeverCanonicalUnlessDependent;
def DeducedType : Type<1>;
def AutoType : DerivedType<DeducedType>;
def DeducedTemplateSpecializationType : DerivedType<DeducedType>;
def InjectedClassNameType : Type, AlwaysDependent, LeafType;
def DependentNameType : Type, AlwaysDependent;
def DependentTemplateSpecializationType : Type, AlwaysDependent;
def PackExpansionType : Type, NeverCanonicalUnlessDependent;
def ObjCTypeParamType : Type, NeverCanonical;
def ObjCObjectType : Type;
def ObjCInterfaceType : DerivedType<ObjCObjectType>, LeafType;
def ObjCObjectPointerType : Type;
def PipeType : Type;
def AtomicType : Type;
