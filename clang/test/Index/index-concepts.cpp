// RUN: c-index-test -test-load-source all %s -std=gnu++20 -fno-delayed-template-parsing | FileCheck %s

template<class T>
struct type_trait {
    const static bool value = false;
};

template<>
struct type_trait<int> {
    const static bool value = true;
};

template <class T>
requires (type_trait<T>::value)
// CHECK: index-concepts.cpp:[[@LINE-1]]:10: ParenExpr= Extent=[[[@LINE-1]]:10 - [[@LINE-1]]:32]
// CHECK: index-concepts.cpp:[[@LINE-2]]:11: DeclRefExpr= Extent=[[[@LINE-2]]:11 - [[@LINE-2]]:31]
// CHECK: index-concepts.cpp:[[@LINE-3]]:11: TemplateRef=type_trait:4:8 Extent=[[[@LINE-3]]:11 - [[@LINE-3]]:21]
// CHECK: index-concepts.cpp:[[@LINE-4]]:22: TypeRef=T:13:17 Extent=[[[@LINE-4]]:22 - [[@LINE-4]]:23]
void indexRequiresClause() {
}

template<class T>
requires (type_trait<T>::value)
// CHECK: index-concepts.cpp:[[@LINE-1]]:10: ParenExpr= Extent=[[[@LINE-1]]:10 - [[@LINE-1]]:32]
// CHECK: index-concepts.cpp:[[@LINE-2]]:11: DeclRefExpr= Extent=[[[@LINE-2]]:11 - [[@LINE-2]]:31]
// CHECK: index-concepts.cpp:[[@LINE-3]]:11: TemplateRef=type_trait:4:8 Extent=[[[@LINE-3]]:11 - [[@LINE-3]]:21]
// CHECK: index-concepts.cpp:[[@LINE-4]]:22: TypeRef=T:22:16 Extent=[[[@LINE-4]]:22 - [[@LINE-4]]:23]
class IndexRequiresClauseInClass {};

template <class T>
concept Con1 = type_trait<T>::value;
// CHECK: index-concepts.cpp:[[@LINE-1]]:9: ConceptDecl=Con1:[[@LINE-1]]:9 (Definition) Extent=[[[@LINE-2]]:1 - [[@LINE-1]]:36]
// CHECK: index-concepts.cpp:[[@LINE-3]]:17: TemplateTypeParameter=T:[[@LINE-3]]:17 (Definition) Extent=[[[@LINE-3]]:11 - [[@LINE-3]]:18] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-3]]:16: DeclRefExpr= Extent=[[[@LINE-3]]:16 - [[@LINE-3]]:36]
// CHECK: index-concepts.cpp:[[@LINE-4]]:16: TemplateRef=type_trait:4:8 Extent=[[[@LINE-4]]:16 - [[@LINE-4]]:26]
// CHECK: index-concepts.cpp:[[@LINE-5]]:27: TypeRef=T:30:17 Extent=[[[@LINE-5]]:27 - [[@LINE-5]]:28]

constexpr int sizeFunc() { return 4; }

template <class T>
concept ConWithLogicalAnd = Con1<T> && sizeof(T) > sizeFunc();
// CHECK: index-concepts.cpp:[[@LINE-1]]:9: ConceptDecl=ConWithLogicalAnd:[[@LINE-1]]:9 (Definition) Extent=[[[@LINE-2]]:1 - [[@LINE-1]]:62]
// CHECK: index-concepts.cpp:[[@LINE-3]]:17: TemplateTypeParameter=T:[[@LINE-3]]:17 (Definition) Extent=[[[@LINE-3]]:11 - [[@LINE-3]]:18] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-3]]:29: BinaryOperator= Extent=[[[@LINE-3]]:29 - [[@LINE-3]]:62]
// CHECK: index-concepts.cpp:[[@LINE-4]]:29: ConceptSpecializationExpr= Extent=[[[@LINE-4]]:29 - [[@LINE-4]]:36]
// CHECK: index-concepts.cpp:[[@LINE-5]]:29: TemplateRef=Con1:31:9 Extent=[[[@LINE-5]]:29 - [[@LINE-5]]:33]
// CHECK: index-concepts.cpp:[[@LINE-6]]:40: BinaryOperator= Extent=[[[@LINE-6]]:40 - [[@LINE-6]]:62]
// CHECK: index-concepts.cpp:[[@LINE-7]]:40: UnaryExpr= Extent=[[[@LINE-7]]:40 - [[@LINE-7]]:49]
// CHECK: index-concepts.cpp:[[@LINE-8]]:47: TypeRef=T:40:17 Extent=[[[@LINE-8]]:47 - [[@LINE-8]]:48]
// CHECK: index-concepts.cpp:[[@LINE-9]]:52: UnexposedExpr=sizeFunc:38:15 Extent=[[[@LINE-9]]:52 - [[@LINE-9]]:62]
// CHECK: index-concepts.cpp:[[@LINE-10]]:52: CallExpr=sizeFunc:38:15 Extent=[[[@LINE-10]]:52 - [[@LINE-10]]:62]
// CHECK: index-concepts.cpp:[[@LINE-11]]:52: UnexposedExpr=sizeFunc:38:15 Extent=[[[@LINE-11]]:52 - [[@LINE-11]]:60]
// CHECK: index-concepts.cpp:[[@LINE-12]]:52: DeclRefExpr=sizeFunc:38:15 Extent=[[[@LINE-12]]:52 - [[@LINE-12]]:60]

namespace ns {

template <class T>
concept ConInNamespace = sizeof(T) > 4;

}

template <class T1, class T2>
concept ConTwoTemplateParams = ns::ConInNamespace<T1> && ConWithLogicalAnd<T2>;
// CHECK: index-concepts.cpp:[[@LINE-1]]:9: ConceptDecl=ConTwoTemplateParams:[[@LINE-1]]:9 (Definition) Extent=[[[@LINE-2]]:1 - [[@LINE-1]]:79]
// CHECK: index-concepts.cpp:[[@LINE-3]]:17: TemplateTypeParameter=T1:[[@LINE-3]]:17 (Definition) Extent=[[[@LINE-3]]:11 - [[@LINE-3]]:19] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-4]]:27: TemplateTypeParameter=T2:[[@LINE-4]]:27 (Definition) Extent=[[[@LINE-4]]:21 - [[@LINE-4]]:29] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-4]]:32: BinaryOperator= Extent=[[[@LINE-4]]:32 - [[@LINE-4]]:79]
// CHECK: index-concepts.cpp:[[@LINE-5]]:32: ConceptSpecializationExpr= Extent=[[[@LINE-5]]:32 - [[@LINE-5]]:54]
// CHECK: index-concepts.cpp:[[@LINE-6]]:32: NamespaceRef=ns:55:11 Extent=[[[@LINE-6]]:32 - [[@LINE-6]]:34]
// CHECK: index-concepts.cpp:[[@LINE-7]]:36: TemplateRef=ConInNamespace:58:9 Extent=[[[@LINE-7]]:36 - [[@LINE-7]]:50]
// CHECK: index-concepts.cpp:[[@LINE-8]]:58: ConceptSpecializationExpr= Extent=[[[@LINE-8]]:58 - [[@LINE-8]]:79]
// CHECK: index-concepts.cpp:[[@LINE-9]]:58: TemplateRef=ConWithLogicalAnd:41:9 Extent=[[[@LINE-9]]:58 - [[@LINE-9]]:75]


struct ConcreteType {};

template<class T>
requires ConTwoTemplateParams<T, ConcreteType>
struct UsesConceptInRequires {};
// CHECK: index-concepts.cpp:[[@LINE-1]]:8: ClassTemplate=UsesConceptInRequires:[[@LINE-1]]:8 (Definition) Extent=[[[@LINE-3]]:1 - [[@LINE-1]]:32]
// CHECK: index-concepts.cpp:[[@LINE-4]]:16: TemplateTypeParameter=T:[[@LINE-4]]:16 (Definition) Extent=[[[@LINE-4]]:10 - [[@LINE-4]]:17] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-4]]:10: ConceptSpecializationExpr= Extent=[[[@LINE-4]]:10 - [[@LINE-4]]:47]
// CHECK: index-concepts.cpp:[[@LINE-5]]:10: TemplateRef=ConTwoTemplateParams:63:9 Extent=[[[@LINE-5]]:10 - [[@LINE-5]]:30]
// CHECK: index-concepts.cpp:[[@LINE-6]]:31: TypeRef=T:[[@LINE-7]]:16 Extent=[[[@LINE-6]]:31 - [[@LINE-6]]:32]
// CHECK: index-concepts.cpp:[[@LINE-7]]:34: TypeRef=struct ConcreteType:[[@LINE-10]]:8 Extent=[[[@LINE-7]]:34 - [[@LINE-7]]:46]


template<ConWithLogicalAnd T>
struct UsesConceptInTemplateArg {};
// CHECK: index-concepts.cpp:[[@LINE-1]]:8: ClassTemplate=UsesConceptInTemplateArg:[[@LINE-1]]:8 (Definition) Extent=[[[@LINE-2]]:1 - [[@LINE-1]]:35]
// CHECK: index-concepts.cpp:[[@LINE-3]]:28: TemplateTypeParameter=T:[[@LINE-3]]:28 (Definition) Extent=[[[@LINE-3]]:10 - [[@LINE-3]]:29] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-4]]:10: TemplateRef=ConWithLogicalAnd:41:9 Extent=[[[@LINE-4]]:10 - [[@LINE-4]]:27]

void usesConceptInAutoParam(ns::ConInNamespace auto x) {}
// CHECK: index-concepts.cpp:[[@LINE-1]]:6: FunctionTemplate=usesConceptInAutoParam:[[@LINE-1]]:6 (Definition)
// CHECK: index-concepts.cpp:[[@LINE-2]]:53: ParmDecl=x:[[@LINE-2]]:53 (Definition) Extent=[[[@LINE-2]]:29 - [[@LINE-2]]:54]
// CHECK: index-concepts.cpp:[[@LINE-3]]:29: NamespaceRef=ns:55:11 Extent=[[[@LINE-3]]:29 - [[@LINE-3]]:31]
// CHECK: index-concepts.cpp:[[@LINE-4]]:33: TemplateRef=ConInNamespace:58:9 Extent=[[[@LINE-4]]:33 - [[@LINE-4]]:47]
// CHECK: index-concepts.cpp:[[@LINE-5]]:48: TypeRef=ns::ConInNamespace auto:[[@LINE-5]]:53 Extent=[[[@LINE-5]]:48 - [[@LINE-5]]:52]
// CHECK: index-concepts.cpp:[[@LINE-6]]:56: CompoundStmt= Extent=[[[@LINE-6]]:56 - [[@LINE-6]]:58]


template<class T>
void testTrailingRequires(const T &x)
requires ns::ConInNamespace<T> && ConTwoTemplateParams<T, ConcreteType> {}
// CHECK: index-concepts.cpp:[[@LINE-2]]:6: FunctionTemplate=testTrailingRequires:[[@LINE-2]]:6 (Definition) Extent=[[[@LINE-3]]:1 - [[@LINE-1]]:75]
// CHECK: index-concepts.cpp:[[@LINE-4]]:16: TemplateTypeParameter=T:[[@LINE-4]]:16 (Definition) Extent=[[[@LINE-4]]:10 - [[@LINE-4]]:17] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-4]]:36: ParmDecl=x:[[@LINE-4]]:36 (Definition) Extent=[[[@LINE-4]]:27 - [[@LINE-4]]:37]
// CHECK: index-concepts.cpp:[[@LINE-5]]:33: TypeRef=T:[[@LINE-6]]:16 Extent=[[[@LINE-5]]:33 - [[@LINE-5]]:34]
// CHECK: index-concepts.cpp:[[@LINE-5]]:10: ConceptSpecializationExpr= Extent=[[[@LINE-5]]:10 - [[@LINE-5]]:31]
// CHECK: index-concepts.cpp:[[@LINE-6]]:10: NamespaceRef=ns:55:11 Extent=[[[@LINE-6]]:10 - [[@LINE-6]]:12]
// CHECK: index-concepts.cpp:[[@LINE-7]]:14: TemplateRef=ConInNamespace:58:9 Extent=[[[@LINE-7]]:14 - [[@LINE-7]]:28]
// CHECK: index-concepts.cpp:[[@LINE-8]]:29: TypeRef=T:[[@LINE-10]]:16 Extent=[[[@LINE-8]]:29 - [[@LINE-8]]:30]
// CHECK: index-concepts.cpp:[[@LINE-9]]:35: ConceptSpecializationExpr= Extent=[[[@LINE-9]]:35 - [[@LINE-9]]:72]
// CHECK: index-concepts.cpp:[[@LINE-10]]:35: TemplateRef=ConTwoTemplateParams:63:9 Extent=[[[@LINE-10]]:35 - [[@LINE-10]]:55]
// CHECK: index-concepts.cpp:[[@LINE-11]]:56: TypeRef=T:[[@LINE-13]]:16 Extent=[[[@LINE-11]]:56 - [[@LINE-11]]:57]
// CHECK: index-concepts.cpp:[[@LINE-12]]:59: TypeRef=struct ConcreteType:75:8 Extent=[[[@LINE-12]]:59 - [[@LINE-12]]:71]


void concreteFunc(ConcreteType);

template<class T>
void genericFunc(const T&x);

template<class T>
concept ConWithRequires = requires(const T& x, ConcreteType value) {
  concreteFunc(value);
  genericFunc(x);
};
// CHECK: index-concepts.cpp:[[@LINE-4]]:9: ConceptDecl=ConWithRequires:[[@LINE-4]]:9 (Definition) Extent=[[[@LINE-5]]:1 - [[@LINE-1]]:2]
// CHECK: index-concepts.cpp:[[@LINE-6]]:16: TemplateTypeParameter=T:[[@LINE-6]]:16 (Definition) Extent=[[[@LINE-6]]:10 - [[@LINE-6]]:17] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-6]]:27: RequiresExpr= Extent=[[[@LINE-6]]:27 - [[@LINE-3]]:2]
// CHECK: index-concepts.cpp:[[@LINE-7]]:61: ParmDecl=value:[[@LINE-7]]:61 (Definition) Extent=[[[@LINE-7]]:48 - [[@LINE-7]]:66]
// CHECK: index-concepts.cpp:[[@LINE-8]]:48: TypeRef=struct ConcreteType:75:8 Extent=[[[@LINE-8]]:48 - [[@LINE-8]]:60]
// CHECK: index-concepts.cpp:[[@LINE-9]]:45: ParmDecl=x:[[@LINE-9]]:45 (Definition) Extent=[[[@LINE-9]]:36 - [[@LINE-9]]:46]
// CHECK: index-concepts.cpp:[[@LINE-10]]:42: TypeRef=T:[[@LINE-11]]:16 Extent=[[[@LINE-10]]:42 - [[@LINE-10]]:43]
// CHECK: index-concepts.cpp:[[@LINE-10]]:3: UnexposedExpr=concreteFunc:[[@LINE-17]]:6 Extent=[[[@LINE-10]]:3 - [[@LINE-10]]:15]
// CHECK: index-concepts.cpp:[[@LINE-11]]:3: DeclRefExpr=concreteFunc:[[@LINE-18]]:6 Extent=[[[@LINE-11]]:3 - [[@LINE-11]]:15]
// CHECK: index-concepts.cpp:[[@LINE-12]]:16: CallExpr=ConcreteType:75:8 Extent=[[[@LINE-12]]:16 - [[@LINE-12]]:21]
// CHECK: index-concepts.cpp:[[@LINE-13]]:16: UnexposedExpr=value:[[@LINE-14]]:61 Extent=[[[@LINE-13]]:16 - [[@LINE-13]]:21]
// CHECK: index-concepts.cpp:[[@LINE-14]]:16: DeclRefExpr=value:[[@LINE-15]]:61 Extent=[[[@LINE-14]]:16 - [[@LINE-14]]:21]
// CHECK: index-concepts.cpp:[[@LINE-14]]:3: DeclRefExpr=[[[@LINE-19]]:6] Extent=[[[@LINE-14]]:3 - [[@LINE-14]]:14]
// CHECK: index-concepts.cpp:[[@LINE-15]]:3: OverloadedDeclRef=genericFunc[[[@LINE-20]]:6] Extent=[[[@LINE-15]]:3 - [[@LINE-15]]:14]
// CHECK: index-concepts.cpp:[[@LINE-16]]:15: DeclRefExpr=x:[[@LINE-18]]:45 Extent=[[[@LINE-16]]:15 - [[@LINE-16]]:16]

template<class T>
concept ConWithCompRequires = requires {
  { genericFunc(T()) } -> ns::ConInNamespace;
  { genericFunc(T()) } -> ConTwoTemplateParams<ConcreteType>;
};
// CHECK: index-concepts.cpp:[[@LINE-4]]:9: ConceptDecl=ConWithCompRequires:[[@LINE-4]]:9 (Definition) Extent=[[[@LINE-5]]:1 - [[@LINE-1]]:2]
// CHECK: index-concepts.cpp:[[@LINE-6]]:16: TemplateTypeParameter=T:[[@LINE-6]]:16 (Definition) Extent=[[[@LINE-6]]:10 - [[@LINE-6]]:17] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-6]]:31: RequiresExpr= Extent=[[[@LINE-6]]:31 - [[@LINE-3]]:2]
// CHECK: index-concepts.cpp:[[@LINE-6]]:5: DeclRefExpr=[123:6] Extent=[[[@LINE-6]]:5 - [[@LINE-6]]:16]
// CHECK: index-concepts.cpp:[[@LINE-7]]:5: OverloadedDeclRef=genericFunc[123:6] Extent=[[[@LINE-7]]:5 - [[@LINE-7]]:16]
// CHECK: index-concepts.cpp:[[@LINE-8]]:17: CallExpr= Extent=[[[@LINE-8]]:17 - [[@LINE-8]]:20]
// CHECK: index-concepts.cpp:[[@LINE-9]]:17: TypeRef=T:[[@LINE-11]]:16 Extent=[[[@LINE-9]]:17 - [[@LINE-9]]:18]
// CHECK: index-concepts.cpp:[[@LINE-10]]:27: NamespaceRef=ns:55:11 Extent=[[[@LINE-10]]:27 - [[@LINE-10]]:29]
// CHECK: index-concepts.cpp:[[@LINE-11]]:31: TemplateRef=ConInNamespace:58:9 Extent=[[[@LINE-11]]:31 - [[@LINE-11]]:45]
// CHECK: index-concepts.cpp:[[@LINE-11]]:5: DeclRefExpr=[123:6] Extent=[[[@LINE-11]]:5 - [[@LINE-11]]:16]
// CHECK: index-concepts.cpp:[[@LINE-12]]:5: OverloadedDeclRef=genericFunc[123:6] Extent=[[[@LINE-12]]:5 - [[@LINE-12]]:16]
// CHECK: index-concepts.cpp:[[@LINE-13]]:17: CallExpr= Extent=[[[@LINE-13]]:17 - [[@LINE-13]]:20]
// CHECK: index-concepts.cpp:[[@LINE-14]]:17: TypeRef=T:[[@LINE-17]]:16 Extent=[[[@LINE-14]]:17 - [[@LINE-14]]:18]
// CHECK: index-concepts.cpp:[[@LINE-15]]:27: TemplateRef=ConTwoTemplateParams:63:9 Extent=[[[@LINE-15]]:27 - [[@LINE-15]]:47]
// CHECK: index-concepts.cpp:[[@LINE-16]]:48: TypeRef=struct ConcreteType:75:8 Extent=[[[@LINE-16]]:48 - [[@LINE-16]]:60]

template<class T>
concept ConWithTypeReq = requires {
  typename type_trait<T>;
};
// CHECK: index-concepts.cpp:[[@LINE-3]]:9: ConceptDecl=ConWithTypeReq:[[@LINE-3]]:9 (Definition) Extent=[[[@LINE-4]]:1 - [[@LINE-1]]:2]
// CHECK: index-concepts.cpp:[[@LINE-5]]:16: TemplateTypeParameter=T:[[@LINE-5]]:16 (Definition) Extent=[[[@LINE-5]]:10 - [[@LINE-5]]:17] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-5]]:26: RequiresExpr= Extent=[[[@LINE-5]]:26 - [[@LINE-3]]:2]
// CHECK: index-concepts.cpp:[[@LINE-5]]:12: TemplateRef=type_trait:4:8 Extent=[[[@LINE-5]]:12 - [[@LINE-5]]:22]
// CHECK: index-concepts.cpp:[[@LINE-6]]:23: TypeRef=T:[[@LINE-8]]:16 Extent=[[[@LINE-6]]:23 - [[@LINE-6]]:24]

template<class T>
concept ConWithNestedReq = requires {
  requires ns::ConInNamespace<T>;
};
// CHECK: index-concepts.cpp:[[@LINE-3]]:9: ConceptDecl=ConWithNestedReq:[[@LINE-3]]:9 (Definition) Extent=[[[@LINE-4]]:1 - [[@LINE-1]]:2]
// CHECK: index-concepts.cpp:[[@LINE-5]]:16: TemplateTypeParameter=T:[[@LINE-5]]:16 (Definition) Extent=[[[@LINE-5]]:10 - [[@LINE-5]]:17] [access=public]
// CHECK: index-concepts.cpp:[[@LINE-5]]:28: RequiresExpr= Extent=[[[@LINE-5]]:28 - [[@LINE-3]]:2]
// CHECK: index-concepts.cpp:[[@LINE-5]]:12: NamespaceRef=ns:55:11 Extent=[[[@LINE-5]]:12 - [[@LINE-5]]:14]
// CHECK: index-concepts.cpp:[[@LINE-6]]:16: TemplateRef=ConInNamespace:58:9 Extent=[[[@LINE-6]]:16 - [[@LINE-6]]:30]
// CHECK: index-concepts.cpp:[[@LINE-7]]:31: TypeRef=T:[[@LINE-9]]:16 Extent=[[[@LINE-7]]:31 - [[@LINE-7]]:32]
