//==--- PropertiesBase.td - Baseline definitions for AST properties -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class ASTNode;

/// The type of the property.
class PropertyType<string typeName = ""> {
	/// The C++ type name for the type.
	string CXXName = !if(!ne(typeName, ""), typeName, NAME);

	/// Whether the C++ type should generally be passed around by reference.
	bit PassByReference = 0;

	/// Whether `const` should be prepended to the type when writing.
	bit ConstWhenWriting = 0;

	/// Given a value of type Optional<CXXName> bound as 'value', yield a
	/// CXXName that can be serialized into a DataStreamTypeWriter.
	string PackOptional = "";

	/// Given a value of type CXXName bound as 'value' that was deserialized
	/// by a DataStreamTypeReader, yield an Optional<CXXName>.
	string UnpackOptional = "";

	/// A list of types for which buffeers must be passed to the read
	/// operations.
	list<PropertyType> BufferElementTypes = [];
}

/// Property types that correspond to specific C++ enums.
class EnumPropertyType<string typeName = ""> : PropertyType<typeName> {}

/// Property types that correspond to a specific C++ class.
/// Supports optional values by using the null representation.
class RefPropertyType<string className> : PropertyType<className # "*"> {
	let PackOptional =
		"value ? *value : nullptr";
	let UnpackOptional =
		"value ? llvm::Optional<" # CXXName # ">(value) : llvm::None";
}

/// Property types that correspond to a specific subclass of another type.
class SubclassPropertyType<string className, PropertyType base>
		: RefPropertyType<className> {
	PropertyType Base = base;
	string SubclassName = className;
	let ConstWhenWriting = base.ConstWhenWriting;
}

/// Property types that support optional values by using their
/// default value.
class DefaultValuePropertyType<string typeName = ""> : PropertyType<typeName> {
	let PackOptional =
		"value ? *value : " # CXXName # "()";
	let UnpackOptional =
		"value.isNull() ? llvm::None : llvm::Optional<" # CXXName # ">(value)";
}

/// Property types that correspond to integer types and support optional
/// values by shifting the value over by 1.
class CountPropertyType<string typeName = ""> : PropertyType<typeName> {
	let PackOptional =
		"value ? *value + 1 : 0";
	let UnpackOptional =
		"value ? llvm::Optional<" # CXXName # ">(value - 1) : llvm::None";
}

def APInt : PropertyType<"llvm::APInt"> { let PassByReference = 1; }
def APSInt : PropertyType<"llvm::APSInt"> { let PassByReference = 1; }
def ArraySizeModifier : EnumPropertyType<"ArrayType::ArraySizeModifier">;
def AttrKind : EnumPropertyType<"attr::Kind">;
def AutoTypeKeyword : EnumPropertyType;
def Bool : PropertyType<"bool">;
def BuiltinTypeKind : EnumPropertyType<"BuiltinType::Kind">;
def CallingConv : EnumPropertyType;
def DeclarationName : PropertyType;
def DeclarationNameKind : EnumPropertyType<"DeclarationName::NameKind">;
def DeclRef : RefPropertyType<"Decl"> { let ConstWhenWriting = 1; }
	def CXXRecordDeclRef :
		SubclassPropertyType<"CXXRecordDecl", DeclRef>;
	def FunctionDeclRef :
		SubclassPropertyType<"FunctionDecl", DeclRef>;
	def NamedDeclRef :
		SubclassPropertyType<"NamedDecl", DeclRef>;
	def NamespaceDeclRef :
		SubclassPropertyType<"NamespaceDecl", DeclRef>;
	def NamespaceAliasDeclRef :
		SubclassPropertyType<"NamespaceAliasDecl", DeclRef>;
	def ObjCProtocolDeclRef :
		SubclassPropertyType<"ObjCProtocolDecl", DeclRef>;
	def ObjCTypeParamDeclRef :
		SubclassPropertyType<"ObjCTypeParamDecl", DeclRef>;
	def TagDeclRef :
		SubclassPropertyType<"TagDecl", DeclRef>;
	def TemplateDeclRef :
		SubclassPropertyType<"TemplateDecl", DeclRef>;
	def TemplateTypeParmDeclRef :
		SubclassPropertyType<"TemplateTypeParmDecl", DeclRef>;
	def TemplateTemplateParmDeclRef :
		SubclassPropertyType<"TemplateTemplateParmDecl", DeclRef>;
	def ValueDeclRef :
		SubclassPropertyType<"ValueDecl", DeclRef>;
def ElaboratedTypeKeyword : EnumPropertyType;
def ExtParameterInfo : PropertyType<"FunctionProtoType::ExtParameterInfo">;
def Identifier : PropertyType<"IdentifierInfo*">;
def NestedNameSpecifier : PropertyType<"NestedNameSpecifier *">;
def NestedNameSpecifierKind :
	EnumPropertyType<"NestedNameSpecifier::SpecifierKind">;
def OverloadedOperatorKind : EnumPropertyType;
def Qualifiers : PropertyType;
def QualType : DefaultValuePropertyType;
def RefQualifierKind : EnumPropertyType;
def Selector : PropertyType;
def SourceLocation : PropertyType;
def StmtRef : RefPropertyType<"Stmt"> { let ConstWhenWriting = 1; }
	def ExprRef : SubclassPropertyType<"Expr", StmtRef>;
def TemplateArgument : PropertyType;
def TemplateArgumentKind : EnumPropertyType<"TemplateArgument::ArgKind">;
def TemplateName : DefaultValuePropertyType;
def TemplateNameKind : EnumPropertyType<"TemplateName::NameKind">;
def UInt32 : CountPropertyType<"uint32_t">;
def UInt64 : CountPropertyType<"uint64_t">;
def UnaryTypeTransformKind : EnumPropertyType<"UnaryTransformType::UTTKind">;
def VectorKind : EnumPropertyType<"VectorType::VectorKind">;

def ExceptionSpecInfo : PropertyType<"FunctionProtoType::ExceptionSpecInfo"> {
	let BufferElementTypes = [ QualType ];
}

/// Arrays.  The corresponding C++ type is ArrayRef of the corresponding
/// C++ type of the element.
class Array<PropertyType element> : PropertyType {
	PropertyType Element = element;
	let BufferElementTypes = [ element ];
}

/// llvm::Optional<T>.  The corresponding C++ type is generally just the
/// corresponding C++ type of the element.
///
/// Optional<Unsigned> may restrict the range of the operand for some
/// serialization clients.
class Optional<PropertyType element> : PropertyType {
	PropertyType Element = element;
	let PassByReference = element.PassByReference;
}

/// A property of an AST node.
class Property<string name, PropertyType type> {
	ASTNode Class;
	string Name = name;
	PropertyType Type = type;

	/// A function for reading the property, expressed in terms of a variable
	/// "node".
	code Read;
}

/// A rule for creating objects of this type.
class Creator<code create> {
	ASTNode Class;

	/// A function for creating values of this kind, expressed in terms of a
	/// variable `ctx` of type `ASTContext &`.  Must also refer to all of the
	/// properties by name.
	code Create = create;
}

/// A rule which overrides some of the normal rules.
class Override {
	ASTNode Class;

	/// Properties from base classes that should be ignored.
	list<string> IgnoredProperties = [];
}
