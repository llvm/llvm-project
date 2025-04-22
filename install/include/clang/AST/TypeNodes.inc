/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* An x-macro database of Clang type nodes                                    *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: TypeNodes.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ABSTRACT_TYPE
#  define ABSTRACT_TYPE(Class, Base) TYPE(Class, Base)
#endif
#ifndef NON_CANONICAL_TYPE
#  define NON_CANONICAL_TYPE(Class, Base) TYPE(Class, Base)
#endif
#ifndef DEPENDENT_TYPE
#  define DEPENDENT_TYPE(Class, Base) TYPE(Class, Base)
#endif
#ifndef NON_CANONICAL_UNLESS_DEPENDENT_TYPE
#  define NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Class, Base) TYPE(Class, Base)
#endif
NON_CANONICAL_TYPE(Adjusted, Type)
NON_CANONICAL_TYPE(Decayed, AdjustedType)
ABSTRACT_TYPE(Array, Type)
TYPE(ConstantArray, ArrayType)
TYPE(ArrayParameter, ConstantArrayType)
DEPENDENT_TYPE(DependentSizedArray, ArrayType)
TYPE(IncompleteArray, ArrayType)
TYPE(VariableArray, ArrayType)
TYPE(Atomic, Type)
NON_CANONICAL_TYPE(Attributed, Type)
NON_CANONICAL_TYPE(BTFTagAttributed, Type)
TYPE(BitInt, Type)
TYPE(BlockPointer, Type)
ABSTRACT_TYPE(BoundsAttributed, Type)
NON_CANONICAL_TYPE(CountAttributed, BoundsAttributedType)
TYPE(Builtin, Type)
TYPE(Complex, Type)
NON_CANONICAL_UNLESS_DEPENDENT_TYPE(Decltype, Type)
ABSTRACT_TYPE(Deduced, Type)
TYPE(Auto, DeducedType)
TYPE(DeducedTemplateSpecialization, DeducedType)
DEPENDENT_TYPE(DependentAddressSpace, Type)
DEPENDENT_TYPE(DependentBitInt, Type)
DEPENDENT_TYPE(DependentName, Type)
DEPENDENT_TYPE(DependentSizedExtVector, Type)
DEPENDENT_TYPE(DependentTemplateSpecialization, Type)
DEPENDENT_TYPE(DependentVector, Type)
NON_CANONICAL_TYPE(Elaborated, Type)
ABSTRACT_TYPE(Function, Type)
TYPE(FunctionNoProto, FunctionType)
TYPE(FunctionProto, FunctionType)
TYPE(HLSLAttributedResource, Type)
DEPENDENT_TYPE(InjectedClassName, Type)
NON_CANONICAL_TYPE(MacroQualified, Type)
ABSTRACT_TYPE(Matrix, Type)
TYPE(ConstantMatrix, MatrixType)
DEPENDENT_TYPE(DependentSizedMatrix, MatrixType)
TYPE(MemberPointer, Type)
TYPE(ObjCObjectPointer, Type)
TYPE(ObjCObject, Type)
TYPE(ObjCInterface, ObjCObjectType)
NON_CANONICAL_TYPE(ObjCTypeParam, Type)
DEPENDENT_TYPE(PackExpansion, Type)
NON_CANONICAL_UNLESS_DEPENDENT_TYPE(PackIndexing, Type)
NON_CANONICAL_TYPE(Paren, Type)
TYPE(Pipe, Type)
TYPE(Pointer, Type)
ABSTRACT_TYPE(Reference, Type)
TYPE(LValueReference, ReferenceType)
TYPE(RValueReference, ReferenceType)
DEPENDENT_TYPE(SubstTemplateTypeParmPack, Type)
NON_CANONICAL_TYPE(SubstTemplateTypeParm, Type)
ABSTRACT_TYPE(Tag, Type)
TYPE(Enum, TagType)
TYPE(Record, TagType)
NON_CANONICAL_UNLESS_DEPENDENT_TYPE(TemplateSpecialization, Type)
DEPENDENT_TYPE(TemplateTypeParm, Type)
NON_CANONICAL_UNLESS_DEPENDENT_TYPE(TypeOfExpr, Type)
NON_CANONICAL_UNLESS_DEPENDENT_TYPE(TypeOf, Type)
NON_CANONICAL_TYPE(Typedef, Type)
NON_CANONICAL_UNLESS_DEPENDENT_TYPE(UnaryTransform, Type)
DEPENDENT_TYPE(UnresolvedUsing, Type)
NON_CANONICAL_TYPE(Using, Type)
TYPE(Vector, Type)
TYPE(ExtVector, VectorType)
#ifdef LAST_TYPE
LAST_TYPE(ExtVector)
#undef LAST_TYPE
#endif
#ifdef LEAF_TYPE
LEAF_TYPE(Builtin)
LEAF_TYPE(Enum)
LEAF_TYPE(InjectedClassName)
LEAF_TYPE(ObjCInterface)
LEAF_TYPE(Record)
LEAF_TYPE(TemplateTypeParm)
#undef LEAF_TYPE
#endif
#undef TYPE
#undef ABSTRACT_TYPE
#undef ABSTRACT_TYPE
#undef NON_CANONICAL_TYPE
#undef DEPENDENT_TYPE
#undef NON_CANONICAL_UNLESS_DEPENDENT_TYPE
