// RUN: %clang_cc1 -Wdocumentation -ast-dump=json -x hlsl -triple dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=JSON
// RUN: %clang_cc1 -Wdocumentation -ast-dump -x hlsl -triple dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefix=AST

// JSON:"kind": "HLSLBufferDecl",
// JSON:"name": "A",
// JSON-NEXT:"bufferKind": "cbuffer",
// JSON:"kind": "TextComment",
// JSON:"text": " CBuffer decl."

/// CBuffer decl.
cbuffer A {
    // JSON: "kind": "VarDecl",
    // JSON: "name": "a",
    // JSON: "qualType": "hlsl_constant float"
    float a;
    // JSON: "kind": "VarDecl",
    // JSON: "name": "b",
    // JSON: "qualType": "hlsl_constant int"
    int b;
}

// AST: TranslationUnitDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: |-NamespaceDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit hlsl
// AST-NEXT: | |-TypeAliasTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit vector
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> class depth 0 index 0 element
// AST-NEXT: | | | `-TemplateArgument type 'float'
// AST-NEXT: | | |   `-typeDetails: BuiltinType {{.*}} 'float'
// AST-NEXT: | | |-NonTypeTemplateParmDecl {{.*}} <<invalid sloc>> <invalid sloc> 'int' depth 0 index 1 element_count
// AST-NEXT: | | | `-TemplateArgument expr '4'
// AST-NEXT: | | |   `-IntegerLiteral {{.*}} <<invalid sloc>> 'int' 4
// AST-NEXT: | | `-TypeAliasDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit vector 'vector<element, element_count>'
// AST-NEXT: | |   `-typeDetails: DependentSizedExtVectorType {{.*}} 'vector<element, element_count>' dependent <invalid sloc>
// AST-NEXT: | |     |-typeDetails: TemplateTypeParmType {{.*}} 'element' dependent depth 0 index 0
// AST-NEXT: | |     | `-TemplateTypeParm {{.*}} 'element'
// AST-NEXT: | |     `-DeclRefExpr {{.*}} <<invalid sloc>> 'int' lvalue NonTypeTemplateParm {{.*}} 'element_count' 'int'
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit Buffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_typed_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class Buffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RWBuffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_typed_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class RWBuffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RasterizerOrderedBuffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_structured_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class RasterizerOrderedBuffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit StructuredBuffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_structured_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class StructuredBuffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RWStructuredBuffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_structured_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class RWStructuredBuffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit AppendStructuredBuffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_structured_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class AppendStructuredBuffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit ConsumeStructuredBuffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_structured_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class ConsumeStructuredBuffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-ClassTemplateDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit RasterizerOrderedStructuredBuffer
// AST-NEXT: | | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> typename depth 0 index 0 element_type
// AST-NEXT: | | |-ConceptSpecializationExpr {{.*}} <<invalid sloc>> 'bool' Concept {{.*}} '__is_structured_resource_element_compatible'
// AST-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}} <<invalid sloc>> <invalid sloc>
// AST-NEXT: | | | | `-TemplateArgument type 'type-parameter-0-0'
// AST-NEXT: | | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'type-parameter-0-0' dependent depth 0 index 0
// AST-NEXT: | | | |     `-TemplateTypeParm {{.*}} depth 0 index 0
// AST-NEXT: | | | `-TemplateArgument type 'element_type':'type-parameter-0-0'
// AST-NEXT: | | |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: | | |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: | | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class RasterizerOrderedStructuredBuffer
// AST-NEXT: | |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class ByteAddressBuffer
// AST-NEXT: | | `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | |-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class RWByteAddressBuffer
// AST-NEXT: | | `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: | `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit <undeserialized declarations> class RasterizerOrderedByteAddressBuffer
// AST-NEXT: |   `-attrDetails: FinalAttr {{.*}} <<invalid sloc>> Implicit final
// AST-NEXT: |-ConceptDecl {{.*}} <<invalid sloc>> <invalid sloc> __is_typed_resource_element_compatible
// AST-NEXT: | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> referenced typename depth 0 index 0 element_type
// AST-NEXT: | `-TypeTraitExpr {{.*}} <<invalid sloc>> 'bool' __builtin_hlsl_is_typed_resource_element_compatible
// AST-NEXT: |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: |-ConceptDecl {{.*}} <<invalid sloc>> <invalid sloc> __is_structured_resource_element_compatible
// AST-NEXT: | |-TemplateTypeParmDecl {{.*}} <<invalid sloc>> <invalid sloc> referenced typename depth 0 index 0 element_type
// AST-NEXT: | `-BinaryOperator {{.*}} <<invalid sloc>> 'bool' lvalue '&&'
// AST-NEXT: |   |-UnaryOperator {{.*}} <<invalid sloc>> 'bool' lvalue prefix '!' cannot overflow
// AST-NEXT: |   | `-TypeTraitExpr {{.*}} <<invalid sloc>> 'bool' __builtin_hlsl_is_intangible
// AST-NEXT: |   |   `-typeDetails: TemplateTypeParmType {{.*}} 'element_type' dependent depth 0 index 0
// AST-NEXT: |   |     `-TemplateTypeParm {{.*}} 'element_type'
// AST-NEXT: |   `-BinaryOperator {{.*}} <<invalid sloc>> 'bool' lvalue '>='
// AST-NEXT: |     |-UnaryExprOrTypeTraitExpr {{.*}} <<invalid sloc>> 'bool' sizeof 'element_type'
// AST: |-UsingDirectiveDecl {{.*}} <<invalid sloc>> <invalid sloc> Namespace {{.*}} 'hlsl'
// AST-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __NSConstantString '__NSConstantString_tag'
// AST-NEXT: | `-typeDetails: RecordType {{.*}} '__NSConstantString_tag'
// AST-NEXT: |   `-CXXRecord {{.*}} '__NSConstantString_tag'
// AST-NEXT: |-TypedefDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit __builtin_va_list 'void *'
// AST-NEXT: | `-typeDetails: PointerType {{.*}} 'void *'
// AST-NEXT: |   `-typeDetails: BuiltinType {{.*}} 'void'
// AST-NEXT: `-HLSLBufferDecl {{.*}} cbuffer A
// AST-NEXT:   |-attrDetails: HLSLResourceClassAttr {{.*}} <<invalid sloc>> Implicit CBuffer
// AST-NEXT:   |-attrDetails: HLSLResourceBindingAttr {{.*}} <<invalid sloc>> Implicit "" "0"
// AST-NEXT:   |-FullComment {{.*}} 
// AST-NEXT:   | `-ParagraphComment {{.*}} 
// AST-NEXT:   |   `-TextComment {{.*}} Text=" CBuffer decl."
// AST-NEXT:   |-VarDecl {{.*}} a 'hlsl_constant float'
// AST-NEXT:   | `-qualTypeDetail: QualType {{.*}} 'hlsl_constant float' hlsl_constant
// AST-NEXT:   |   `-typeDetails: BuiltinType {{.*}} 'float'
// AST-NEXT:   |-VarDecl {{.*}} b 'hlsl_constant int'
// AST-NEXT:   | `-qualTypeDetail: QualType {{.*}} 'hlsl_constant int' hlsl_constant
// AST-NEXT:   |   `-typeDetails: BuiltinType {{.*}} 'int'
// AST-NEXT:   `-CXXRecordDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit struct __cblayout_A definition
// AST-NEXT:     |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// AST-NEXT:     | |-DefaultConstructor exists trivial needs_implicit
// AST-NEXT:     | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// AST-NEXT:     | |-MoveConstructor exists simple trivial needs_implicit
// AST-NEXT:     | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// AST-NEXT:     | |-MoveAssignment exists simple trivial needs_implicit
// AST-NEXT:     | `-Destructor simple irrelevant trivial needs_implicit
// AST-NEXT:     |-attrDetails: PackedAttr {{.*}} <<invalid sloc>> Implicit
// AST-NEXT:     |-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> a 'float'
// AST-NEXT:     `-FieldDecl {{.*}} <<invalid sloc>> <invalid sloc> b 'int'

