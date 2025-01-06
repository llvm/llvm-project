// RUN: %clang_cc1 -std=gnu++11 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -std=gnu++11 -ast-dump %s | FileCheck %s

namespace attribute_aligned {
  template<int N>
  struct X {
    char c[1] __attribute__((__aligned__((N)))); // expected-error {{alignment is not a power of 2}}
  };

  template <bool X> struct check {
    int check_failed[X ? 1 : -1]; // expected-error {{array with a negative size}}
  };

  template <int N> struct check_alignment {
    typedef check<N == sizeof(X<N>)> t; // expected-note {{in instantiation}}
  };

  check_alignment<1>::t c1;
  check_alignment<2>::t c2;
  check_alignment<3>::t c3; // expected-note 2 {{in instantiation}}
  check_alignment<4>::t c4;

  template<unsigned Size, unsigned Align>
  class my_aligned_storage
  {
    __attribute__((aligned(Align))) char storage[Size];
  };

  template<typename T>
  class C {
  public:
    C() {
      static_assert(sizeof(t) == sizeof(T), "my_aligned_storage size wrong");
      static_assert(alignof(t) == alignof(T), "my_aligned_storage align wrong"); // expected-warning{{'alignof' applied to an expression is a GNU extension}}
    }

  private:
    my_aligned_storage<sizeof(T), alignof(T)> t;
  };

  C<double> cd;
}

namespace PR9049 {
  extern const void *CFRetain(const void *ref);

  template<typename T> __attribute__((cf_returns_retained))
  inline T WBCFRetain(T aValue) { return aValue ? (T)CFRetain(aValue) : (T)0; }


  extern void CFRelease(const void *ref);

  template<typename T>
  inline void WBCFRelease(__attribute__((cf_consumed)) T aValue) { if(aValue) CFRelease(aValue); }
}

namespace attribute_annotate {
// CHECK: FunctionTemplateDecl {{.*}} HasAnnotations
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_FOO"
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_BAR"
// CHECK: FunctionDecl {{.*}} HasAnnotations
// CHECK:   TemplateArgument type 'int'
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_FOO"
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_BAR"
template<typename T> [[clang::annotate("ANNOTATE_FOO"), clang::annotate("ANNOTATE_BAR")]] void HasAnnotations();
void UseAnnotations() { HasAnnotations<int>(); }

// CHECK: FunctionTemplateDecl {{.*}} HasStmtAnnotations
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_BAZ"
// CHECK: FunctionDecl {{.*}} HasStmtAnnotations
// CHECK:   TemplateArgument type 'int'
// CHECK:   AnnotateAttr {{.*}} "ANNOTATE_BAZ"
template<typename T> void HasStmtAnnotations() {
  int x = 0;
  [[clang::annotate("ANNOTATE_BAZ")]] x++;
}
void UseStmtAnnotations() { HasStmtAnnotations<int>(); }

// CHECK:      FunctionTemplateDecl {{.*}} HasPackAnnotations
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   FunctionDecl {{.*}} HasPackAnnotations 'void ()'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_BAZ"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   FunctionDecl {{.*}} used HasPackAnnotations 'void ()'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_BAZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
template <int... Is> [[clang::annotate("ANNOTATE_BAZ", Is...)]] void HasPackAnnotations();
void UsePackAnnotations() { HasPackAnnotations<1, 2, 3>(); }

// CHECK:      FunctionTemplateDecl {{.*}} HasStmtPackAnnotations
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   FunctionDecl {{.*}} HasStmtPackAnnotations 'void ()'
// CHECK:          AttributedStmt {{.*}}
// CHECK-NEXT:       AnnotateAttr {{.*}} "ANNOTATE_QUUX"
// CHECK-NEXT:         PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:           DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK:        FunctionDecl {{.*}} used HasStmtPackAnnotations 'void ()'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK:          AttributedStmt {{.*}}
// CHECK-NEXT:       AnnotateAttr {{.*}} "ANNOTATE_QUUX"
// CHECK-NEXT:         PackExpansionExpr {{.*}}
// CHECK-NEXT:         SubstNonTypeTemplateParmPackExpr {{.*}}
// CHECK-NEXT:         NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           TemplateArgument pack '<1, 2, 3>'
// CHECK-NEXT:             TemplateArgument integral '1'
// CHECK-NEXT:             TemplateArgument integral '2'
// CHECK-NEXT:             TemplateArgument integral '3'
template <int... Is> void HasStmtPackAnnotations() {
  int x = 0;
  [[clang::annotate("ANNOTATE_QUUX", Is...)]] x++;
}
void UseStmtPackAnnotations() { HasStmtPackAnnotations<1, 2, 3>(); }

template <int... Is> [[clang::annotate(Is...)]] void HasOnlyPackAnnotation() {} // expected-error {{expected string literal as argument of 'annotate' attribute}}

void UseOnlyPackAnnotations() {
  HasOnlyPackAnnotation<>();
  HasOnlyPackAnnotation<1>();
}

// CHECK:      ClassTemplateDecl {{.*}} AnnotatedPackTemplateStruct
// CHECK-NEXT:   TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 1 ... Is
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_FOZ"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument{{.*}} type 'int'
// CHECK-NEXT:       BuiltinType {{.*}} 'int'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_BOO"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument type 'float'
// CHECK-NEXT:       BuiltinType {{.*}} 'float'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_FOZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 4
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 5
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 6
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument type 'bool'
// CHECK-NEXT:       BuiltinType {{.*}} 'bool'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '7'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '8'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '9'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_FOZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 7
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 1 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 7
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 8
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 1 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 8
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 9
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 1 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 9
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument type 'char'
// CHECK-NEXT:       BuiltinType {{.*}} 'char'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument{{.*}} type 'char'
// CHECK-NEXT:       BuiltinType {{.*}} 'char'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
// CHECK-NEXT: ClassTemplatePartialSpecializationDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   TemplateArgument{{.*}} type 'int'
// CHECK-NEXT:     BuiltinType {{.*}} 'int'
// CHECK-NEXT:   TemplateArgument{{.*}} pack
// CHECK-NEXT:     TemplateArgument{{.*}} expr
// CHECK-NEXT:       PackExpansionExpr {{.*}} 'int'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   AnnotateAttr {{.*}} "ANNOTATE_BOO"
// CHECK-NEXT:     PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:       DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
// CHECK-NEXT: ClassTemplatePartialSpecializationDecl {{.*}} struct AnnotatedPackTemplateStruct definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   TemplateArgument{{.*}} type 'float'
// CHECK-NEXT:     BuiltinType {{.*}} 'float'
// CHECK-NEXT:   TemplateArgument{{.*}} pack
// CHECK-NEXT:     TemplateArgument{{.*}} expr
// CHECK-NEXT:       PackExpansionExpr {{.*}} 'int'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   AnnotateAttr {{.*}} "ANNOTATE_FOZ"
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 4
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 5
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT:     ConstantExpr {{.*}} 'int'
// CHECK-NEXT:       value: Int 6
// CHECK-NEXT:       IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct AnnotatedPackTemplateStruct
template <typename T, int... Is> struct [[clang::annotate("ANNOTATE_FOZ", Is...)]] AnnotatedPackTemplateStruct{};
template <int... Is> struct [[clang::annotate("ANNOTATE_BOO", Is...)]] AnnotatedPackTemplateStruct<int, Is...>{};
template <int... Is> struct [[clang::annotate("ANNOTATE_FOZ", 4, 5, 6)]] AnnotatedPackTemplateStruct<float, Is...>{};
template <int... Is> struct [[clang::annotate(Is...)]] AnnotatedPackTemplateStruct<char, Is...>{}; // expected-error {{expected string literal as argument of 'annotate' attribute}}
void UseAnnotatedPackTemplateStructSpecializations() {
  AnnotatedPackTemplateStruct<int, 1, 2, 3> Instance1{};
  AnnotatedPackTemplateStruct<float, 3, 2, 1> Instance2{};
  AnnotatedPackTemplateStruct<bool, 7, 8, 9> Instance3{};
  AnnotatedPackTemplateStruct<char, 1, 2, 3> Instance4{};
  AnnotatedPackTemplateStruct<char> Instance5{};
}

// CHECK:      ClassTemplateDecl {{.*}} InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT:   TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} 'int' depth 0 index 1 ... Is
// CHECK-NEXT:   CXXRecordDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecialization {{.*}} 'InvalidAnnotatedPackTemplateStruct'
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument{{.*}} type 'int'
// CHECK-NEXT:       BuiltinType {{.*}} 'int'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_BIR"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument{{.*}} type 'float'
// CHECK-NEXT:       BuiltinType {{.*}} 'float'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument{{.*}} type 'bool'
// CHECK-NEXT:       BuiltinType {{.*}} 'bool'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '7'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '8'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '9'
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT:   ClassTemplateSpecializationDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:     DefinitionData
// CHECK-NEXT:       DefaultConstructor
// CHECK-NEXT:       CopyConstructor
// CHECK-NEXT:       MoveConstructor
// CHECK-NEXT:       CopyAssignment
// CHECK-NEXT:       MoveAssignment
// CHECK-NEXT:       Destructor
// CHECK-NEXT:     TemplateArgument{{.*}} type 'bool'
// CHECK-NEXT:       BuiltinType {{.*}} 'bool'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:     CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT: ClassTemplatePartialSpecializationDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   TemplateArgument{{.*}} type 'int'
// CHECK-NEXT:     BuiltinType {{.*}} 'int'
// CHECK-NEXT:   TemplateArgument{{.*}} pack
// CHECK-NEXT:     TemplateArgument{{.*}} expr
// CHECK-NEXT:       PackExpansionExpr {{.*}} 'int'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   AnnotateAttr {{.*}} "ANNOTATE_BIR"
// CHECK-NEXT:     PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:       DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT: ClassTemplatePartialSpecializationDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   TemplateArgument{{.*}} type 'float'
// CHECK-NEXT:     BuiltinType {{.*}} 'float'
// CHECK-NEXT:   TemplateArgument{{.*}} pack
// CHECK-NEXT:     TemplateArgument{{.*}} expr
// CHECK-NEXT:       PackExpansionExpr {{.*}} 'int'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
// CHECK-NEXT: ClassTemplateSpecializationDecl {{.*}} struct InvalidAnnotatedPackTemplateStruct definition
// CHECK-NEXT:   DefinitionData
// CHECK-NEXT:     DefaultConstructor
// CHECK-NEXT:     CopyConstructor
// CHECK-NEXT:     MoveConstructor
// CHECK-NEXT:     CopyAssignment
// CHECK-NEXT:     MoveAssignment
// CHECK-NEXT:     Destructor
// CHECK-NEXT:   TemplateArgument{{.*}} type 'char'
// CHECK-NEXT:     BuiltinType {{.*}} 'char'
// CHECK-NEXT:   TemplateArgument{{.*}} pack
// CHECK-NEXT:     TemplateArgument{{.*}} integral '5'
// CHECK-NEXT:     TemplateArgument{{.*}} integral '6'
// CHECK-NEXT:     TemplateArgument{{.*}} integral '7'
// CHECK-NEXT:   CXXRecordDecl {{.*}} implicit struct InvalidAnnotatedPackTemplateStruct
template <typename T, int... Is> struct InvalidAnnotatedPackTemplateStruct{};
template <int... Is> struct [[clang::annotate("ANNOTATE_BIR", Is...)]] InvalidAnnotatedPackTemplateStruct<int, Is...>{};
template <int... Is> struct InvalidAnnotatedPackTemplateStruct<float, Is...> {};
template <> struct InvalidAnnotatedPackTemplateStruct<char, 5, 6, 7> {};
void UseInvalidAnnotatedPackTemplateStruct() {
  InvalidAnnotatedPackTemplateStruct<int, 1, 2, 3> Instance1{};
  InvalidAnnotatedPackTemplateStruct<float, 3, 2, 1> Instance2{};
  InvalidAnnotatedPackTemplateStruct<char, 5, 6, 7> Instance3{};
  InvalidAnnotatedPackTemplateStruct<bool, 7, 8, 9> Instance4{};
  InvalidAnnotatedPackTemplateStruct<bool> Instance5{};
}

// CHECK:      FunctionTemplateDecl {{.*}} RedeclaredAnnotatedFunc
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   FunctionDecl {{.*}} RedeclaredAnnotatedFunc 'void ()'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_FAR"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   FunctionDecl {{.*}} used RedeclaredAnnotatedFunc 'void ()'
// CHECK-NEXT:     TemplateArgument{{.*}} pack
// CHECK-NEXT:       TemplateArgument{{.*}} integral '1'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '2'
// CHECK-NEXT:       TemplateArgument{{.*}} integral '3'
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_FAR"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 1
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 2
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 3
// CHECK-NEXT:         SubstNonTypeTemplateParmExpr {{.*}} 'int'
// CHECK-NEXT:           NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:           IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_FIZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 4
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 5
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_BOZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 6
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT: FunctionTemplateDecl {{.*}} prev {{.*}} RedeclaredAnnotatedFunc
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} referenced 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   FunctionDecl {{.*}} prev {{.*}} RedeclaredAnnotatedFunc 'void ()'
// CHECK-NEXT:     AnnotateAttr {{.*}} Inherited "ANNOTATE_FAR"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_BOZ"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:   Function {{.*}} 'RedeclaredAnnotatedFunc' 'void ()'
// CHECK-NEXT: FunctionTemplateDecl {{.*}} prev {{.*}} RedeclaredAnnotatedFunc
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   FunctionDecl {{.*}} prev {{.*}} RedeclaredAnnotatedFunc 'void ()'
// CHECK-NEXT:     AnnotateAttr {{.*}} Inherited "ANNOTATE_FAR"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:     AnnotateAttr {{.*}} Inherited "ANNOTATE_BOZ"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_FIZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 4
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 5
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT:   Function {{.*}} 'RedeclaredAnnotatedFunc' 'void ()'
// CHECK-NEXT: FunctionTemplateDecl {{.*}} prev {{.*}} RedeclaredAnnotatedFunc
// CHECK-NEXT:   NonTypeTemplateParmDecl {{.*}} 'int' depth 0 index 0 ... Is
// CHECK-NEXT:   FunctionDecl {{.*}} prev {{.*}} RedeclaredAnnotatedFunc 'void ()'
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:     AnnotateAttr {{.*}} Inherited "ANNOTATE_FAR"
// CHECK-NEXT:       PackExpansionExpr {{.*}} '<dependent type>'
// CHECK-NEXT:         DeclRefExpr {{.*}} 'int' NonTypeTemplateParm {{.*}} 'Is' 'int'
// CHECK-NEXT:     AnnotateAttr {{.*}} Inherited "ANNOTATE_FIZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 4
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 5
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT:     AnnotateAttr {{.*}} "ANNOTATE_BOZ"
// CHECK-NEXT:       ConstantExpr {{.*}} 'int'
// CHECK-NEXT:         value: Int 6
// CHECK-NEXT:         IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT:   Function {{.*}} 'RedeclaredAnnotatedFunc' 'void ()'
// CHECK-NEXT: EmptyDecl
template <int... Is> [[clang::annotate("ANNOTATE_FAR", Is...)]] void RedeclaredAnnotatedFunc();
template <int... Is> [[clang::annotate("ANNOTATE_BOZ", Is...)]] void RedeclaredAnnotatedFunc();
template <int... Is> [[clang::annotate("ANNOTATE_FIZ", 4, 5)]] void RedeclaredAnnotatedFunc();
template <int... Is> [[clang::annotate("ANNOTATE_BOZ", 6)]] void RedeclaredAnnotatedFunc(){};
void UseRedeclaredAnnotatedFunc() {
  RedeclaredAnnotatedFunc<1, 2, 3>();
}

} // namespace attribute_annotate

namespace preferred_name {
  int x [[clang::preferred_name("frank")]]; // expected-error {{expected a type}}
  int y [[clang::preferred_name(int)]]; // expected-warning {{'preferred_name' attribute only applies to class templates}}
  struct [[clang::preferred_name(int)]] A; // expected-warning {{'preferred_name' attribute only applies to class templates}}
  template<typename T> struct [[clang::preferred_name(int)]] B; // expected-error {{argument 'int' to 'preferred_name' attribute is not a typedef for a specialization of 'B'}}
  template<typename T> struct C;
  using X = C<int>; // expected-note {{'X' declared here}}
  typedef C<float> Y;
  using Z = const C<double>; // expected-note {{'Z' declared here}}
  template<typename T> struct [[clang::preferred_name(C<int>)]] C; // expected-error {{argument 'C<int>' to 'preferred_name' attribute is not a typedef for a specialization of 'C'}}
  template<typename T> struct [[clang::preferred_name(X), clang::preferred_name(Y)]] C;
  template<typename T> struct [[clang::preferred_name(const X)]] C; // expected-error {{argument 'const X'}}
  template<typename T> struct [[clang::preferred_name(Z)]] C; // expected-error {{argument 'Z' (aka 'const C<double>')}}
  template<typename T> struct C {};

  // CHECK: ClassTemplateDecl {{.*}} <line:[[@LINE-10]]:{{.*}} C
  // CHECK:   ClassTemplateSpecializationDecl {{.*}} struct C definition
  // CHECK:     TemplateArgument type 'int'
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     PreferredNameAttr {{.*}} X
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     CXXRecordDecl
  // CHECK:   ClassTemplateSpecializationDecl {{.*}} struct C definition
  // CHECK:     TemplateArgument type 'float'
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     PreferredNameAttr {{.*}} Y
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     CXXRecordDecl
  // CHECK:   ClassTemplateSpecializationDecl {{.*}} struct C definition
  // CHECK:     TemplateArgument type 'double'
  // CHECK-NOT: PreferredNameAttr
  // CHECK:     CXXRecordDecl

  // Check this doesn't cause us to instantiate the same attribute multiple times.
  C<float> *cf1;
  C<float> *cf2;

  void f(C<int> a, C<float> b, C<double> c) {
    auto p = a;
    auto q = b;
    auto r = c;
    p.f(); // expected-error {{no member named 'f' in 'preferred_name::X'}}
    q.f(); // expected-error {{no member named 'f' in 'preferred_name::Y'}}
    r.f(); // expected-error {{no member named 'f' in 'preferred_name::C<double>'}}
  }

  template<typename T> struct D;
  using DInt = D<int>;
  template<typename T> struct __attribute__((__preferred_name__(DInt))) D {};
  template struct D<int>;
  int use_dint = D<int>().get(); // expected-error {{no member named 'get' in 'preferred_name::DInt'}}

  template<typename T> struct MemberTemplate {
    template<typename U> struct Iter;
    using iterator = Iter<T>;
    using const_iterator = Iter<const T>;
    template<typename U>
    struct [[clang::preferred_name(iterator),
             clang::preferred_name(const_iterator)]] Iter {};
  };
  template<typename T> T desugar(T);
  auto it = desugar(MemberTemplate<int>::Iter<const int>());
  int n = it; // expected-error {{no viable conversion from 'preferred_name::MemberTemplate<int>::const_iterator' to 'int'}}

  template<int A, int B, typename ...T> struct Foo;
  template<typename ...T> using Bar = Foo<1, 2, T...>;
  template<int A, int B, typename ...T> struct [[clang::preferred_name(::preferred_name::Bar<T...>)]] Foo {};
  Foo<1, 2, int, float>::nosuch x; // expected-error {{no type named 'nosuch' in 'preferred_name::Bar<int, float>'}}
}
::preferred_name::Foo<1, 2, int, float>::nosuch x; // expected-error {{no type named 'nosuch' in 'preferred_name::Bar<int, float>'}}
