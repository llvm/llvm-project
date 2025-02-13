// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -ast-dump %s 2>&1 | FileCheck %s --check-prefix=ATTR-ONLY
// FIXME: Compilation fails when return_size bounds check is enabled (rdar://150044760) so disable it for now.
// RUN: %clang_cc1 -fbounds-safety -fexperimental-bounds-safety-cxx  -fno-bounds-safety-bringup-missing-checks=return_size -DBOUNDS_SAFETY -ast-dump %s 2>&1 | FileCheck %s --check-prefix=BOUNDS-CHECK

#include <ptrcheck.h>

void * __sized_by_or_null(size) malloc(int size);

// ATTR-ONLY: |-FunctionDecl {{.*}} malloc 'void * __sized_by_or_null(size)(int)'
// BOUNDS-CHECK: |-FunctionDecl {{.*}} malloc 'void *__single __sized_by_or_null(size)(int)'

template <typename T>
T * __sized_by_or_null(sizeof(T) * n) mymalloc(int n) {
    return static_cast<T * __sized_by_or_null(4 * n)>(malloc(sizeof(T) * n));
}

// ATTR-ONLY: |-FunctionTemplateDecl {{.*}} mymalloc
// ATTR-ONLY: | |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// ATTR-ONLY: | |-FunctionDecl {{.*}} mymalloc 'T *(int)'
// ATTR-ONLY: | | | `-ReturnStmt
// FIXME: late parse attributes when applied to types rdar://143865865
// ATTR-ONLY: | | |   `-CXXStaticCastExpr {{.*}} 'T *' static_cast<T *> <Dependent>
// ATTR-ONLY: | | |     `-CallExpr {{.*}} 'void * __sized_by_or_null(size)':'void *'
// ATTR-ONLY: | | `-SizedByOrNullAttr
// ATTR-ONLY: | |   `-BinaryOperator {{.*}} '*'
// ATTR-ONLY: | |     |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// ATTR-ONLY: | |       `-DeclRefExpr {{.*}} 'n' 'int'

// BOUNDS-CHECK: |-FunctionTemplateDecl {{.*}} mymalloc
// BOUNDS-CHECK: | |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// BOUNDS-CHECK: | |-FunctionDecl {{.*}} mymalloc 'T *__single(int)'
// BOUNDS-CHECK: | | | `-ReturnStmt
// BOUNDS-CHECK: | | |   `-CXXStaticCastExpr {{.*}} 'T *' static_cast<T *> <Dependent>
// BOUNDS-CHECK: | | |     `-CallExpr {{.*}} 'void *__single  __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: | | `-SizedByOrNullAttr
// BOUNDS-CHECK: | |   `-BinaryOperator {{.*}} '*'
// BOUNDS-CHECK: | |     |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// BOUNDS-CHECK: | |       `-DeclRefExpr {{.*}} 'n' 'int'

// ATTR-ONLY: | `-FunctionDecl {{.*}} mymalloc 'int * __sized_by_or_null(4UL * n)(int)'
// ATTR-ONLY: |   |-TemplateArgument type 'int'
// ATTR-ONLY: |       `-CXXStaticCastExpr {{.*}} 'int *' static_cast<int *> <BitCast>
// ATTR-ONLY: |         `-CallExpr {{.*}} 'void * __sized_by_or_null(size)':'void *'
// ATTR-ONLY: |             `-BinaryOperator {{.*}} '*'
// ATTR-ONLY: |               |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'int'
// ATTR-ONLY: |                   `-DeclRefExpr {{.*}} 'n' 'int'

// BOUNDS-CHECK: | `-FunctionDecl {{.*}} mymalloc 'int *__single __sized_by_or_null(4UL * n)(int)'
// BOUNDS-CHECK: |   |-TemplateArgument type 'int'
// BOUNDS-CHECK: |       `-CXXStaticCastExpr {{.*}} 'int *' static_cast<int *> <BitCast>
// BOUNDS-CHECK: |           `-MaterializeSequenceExpr {{.*}} <Unbind>
// BOUNDS-CHECK: |             |-MaterializeSequenceExpr {{.*}} <Bind>
// BOUNDS-CHECK: |             | |-BoundsSafetyPointerPromotionExpr {{.*}} 'void *__bidi_indexable'
// BOUNDS-CHECK: |             | | |-OpaqueValueExpr {{.*}} 'void *__single __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: |             | | | `-CallExpr {{.*}} 'void *__single __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: |             | | |   `-OpaqueValueExpr {{.*}}
// BOUNDS-CHECK: |             | | |       `-BinaryOperator {{.*}} '*'
// BOUNDS-CHECK: |             | | |         |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'int'
// BOUNDS-CHECK: |             | | |             `-DeclRefExpr {{.*}} 'n' 'int'

template <typename T>
T bar(int x) {
    int y;
    int * __sized_by_or_null(sizeof(T) * y) p;
    p = mymalloc<T>(x);
    y = x;
    return *p;
}

// ATTR-ONLY: |-FunctionTemplateDecl {{.*}} bar
// ATTR-ONLY: | |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// ATTR-ONLY: | |-FunctionDecl {{.*}} bar 'T (int)'
// ATTR-ONLY: | | `-CompoundStmt
// ATTR-ONLY: | |   | `-VarDecl {{.*}} y 'int'
// ATTR-ONLY: | |   | `-VarDecl {{.*}} p 'int *'
// ATTR-ONLY: | |   |   `-SizedByOrNullAttr
// ATTR-ONLY: | |   |     `-BinaryOperator {{.*}} '*'
// ATTR-ONLY: | |   |       |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// ATTR-ONLY: | |   |       `-DeclRefExpr {{.*}} 'y'
// ATTR-ONLY: | |   |-BinaryOperator {{.*}} '<dependent type>' '='
// ATTR-ONLY: | |   | |-DeclRefExpr {{.*}} 'p'
// ATTR-ONLY: | |   | `-CallExpr {{.*}} '<dependent type>'
// ATTR-ONLY: | |   |   |-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (ADL) = 'mymalloc'
// ATTR-ONLY: | |   |   | `-TemplateArgument type 'T':'type-parameter-0-0'
// ATTR-ONLY: | |   |   |   `-TemplateTypeParmType {{.*}} 'T' dependent depth 0 index 0
// ATTR-ONLY: | |   |   |     `-TemplateTypeParm {{.*}} 'T'
// ATTR-ONLY: | |   |   `-DeclRefExpr {{.*}} 'x'
// ATTR-ONLY: | |   |-BinaryOperator {{.*}} '='
// ATTR-ONLY: | |   | |-DeclRefExpr {{.*}} 'y'
// ATTR-ONLY: | |   | `-DeclRefExpr {{.*}} 'x'
// ATTR-ONLY: | |   `-ReturnStmt
// ATTR-ONLY: | |     `-UnaryOperator {{.*}} '*'
// ATTR-ONLY: | |       `-DeclRefExpr {{.*}} 'p' 'int *'

// BOUNDS-CHECK: |-FunctionTemplateDecl {{.*}} bar
// BOUNDS-CHECK: | |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// BOUNDS-CHECK: | |-FunctionDecl {{.*}} bar 'T (int)'
// BOUNDS-CHECK: | | `-CompoundStmt
// BOUNDS-CHECK: | |   | `-VarDecl {{.*}} y 'int'
// BOUNDS-CHECK: | |   | `-VarDecl {{.*}} p 'int *__bidi_indexable'
// BOUNDS-CHECK: | |   |   `-SizedByOrNullAttr
// BOUNDS-CHECK: | |   |     `-BinaryOperator {{.*}} '*'
// BOUNDS-CHECK: | |   |       |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// BOUNDS-CHECK: | |   |           `-DeclRefExpr {{.*}} 'y'
// BOUNDS-CHECK: | |   |-BinaryOperator {{.*}} '<dependent type>' '='
// BOUNDS-CHECK: | |   | |-DeclRefExpr {{.*}} 'p' 'int *__bidi_indexable'
// BOUNDS-CHECK: | |   | `-CallExpr {{.*}} '<dependent type>'
// BOUNDS-CHECK: | |   |   |-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (ADL) = 'mymalloc'
// BOUNDS-CHECK: | |   |   | `-TemplateArgument type 'T':'type-parameter-0-0'
// BOUNDS-CHECK: | |   |   |   `-TemplateTypeParmType {{.*}} 'T'
// BOUNDS-CHECK: | |   |   |     `-TemplateTypeParm {{.*}} 'T'
// BOUNDS-CHECK: | |   |   `-DeclRefExpr {{.*}} 'x' 'int'
// BOUNDS-CHECK: | |   |-BinaryOperator {{.*}} 'int' lvalue '='
// BOUNDS-CHECK: | |   | |-DeclRefExpr {{.*}} 'y'
// BOUNDS-CHECK: | |   |   `-DeclRefExpr {{.*}} 'x'
// BOUNDS-CHECK: | |   `-ReturnStmt
// BOUNDS-CHECK: | |     `-UnaryOperator {{.*}} '*'
// BOUNDS-CHECK: | |         `-DeclRefExpr {{.*}} 'p'


// ATTR-ONLY: | `-FunctionDecl {{.*}} bar 'int (int)'
// ATTR-ONLY: |   |-TemplateArgument type 'int'
// ATTR-ONLY: |   `-CompoundStmt
// ATTR-ONLY: |     | `-VarDecl {{.*}} y
// ATTR-ONLY: |     |   `-DependerDeclsAttr
// ATTR-ONLY: |     | `-VarDecl {{.*}} p 'int * __sized_by_or_null(4UL * y)':'int *'
// ATTR-ONLY: |     |-BinaryOperator {{.*}} '='
// ATTR-ONLY: |     | |-DeclRefExpr {{.*}} 'p' 'int * __sized_by_or_null(4UL * y)':'int *'
// ATTR-ONLY: |     | `-CallExpr
// ATTR-ONLY: |     |   `-DeclRefExpr {{.*}} 'x' 'int'
// ATTR-ONLY: |     |-BinaryOperator {{.*}} '='
// ATTR-ONLY: |     | |-DeclRefExpr {{.*}} 'y' 'int'
// ATTR-ONLY: |     | `-DeclRefExpr {{.*}} 'x' 'int'
// ATTR-ONLY: |     `-ReturnStmt
// ATTR-ONLY: |         `-UnaryOperator {{.*}} '*'
// ATTR-ONLY: |             `-DeclRefExpr {{.*}} 'p' 'int * __sized_by_or_null(4UL * y)':'int *'

// BOUNDS-CHECK: | `-FunctionDecl {{.*}} bar 'int (int)'
// BOUNDS-CHECK: |   |-TemplateArgument type 'int'
// BOUNDS-CHECK: |   `-CompoundStmt
// BOUNDS-CHECK: |     | `-VarDecl {{.*}} y
// BOUNDS-CHECK: |     |   `-DependerDeclsAttr
// BOUNDS-CHECK: |     | `-VarDecl {{.*}} p 'int *__single __sized_by_or_null(4UL * y)':'int *__single'
// BOUNDS-CHECK: |     |-MaterializeSequenceExpr {{.*}} <Bind>
// BOUNDS-CHECK: |     | |-BoundsCheckExpr {{.*}} 'mymalloc<int>(x) <= __builtin_get_pointer_upper_bound(mymalloc<int>(x)) && __builtin_get_pointer_lower_bound(mymalloc<int>(x)) <= mymalloc<int>(x) && !mymalloc<int>(x) || 4UL * x <= (char *)__builtin_get_pointer_upper_bound(mymalloc<int>(x)) - (char *__single)mymalloc<int>(x)'
// BOUNDS-CHECK: |     | | |-BinaryOperator {{.*}} 'int *__single __sized_by_or_null(4UL * y)':'int *__single' lvalue '='
// BOUNDS-CHECK: |     | | | |-DeclRefExpr {{.*}} 'p'
// BOUNDS-CHECK: |     | | | `-OpaqueValueExpr
// BOUNDS-CHECK: |     | | |     `-MaterializeSequenceExpr {{.*}} <Unbind>
// BOUNDS-CHECK: |     | | |       |-MaterializeSequenceExpr {{.*}} <Bind>
// BOUNDS-CHECK: |     | | |       | |-BoundsSafetyPointerPromotionExpr {{.*}} 'int *__bidi_indexable'
// BOUNDS-CHECK: |     | | |       | | |-OpaqueValueExpr {{.*}}
// BOUNDS-CHECK: |     | | |       | | | `-CallExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |     | | |       | | |   `-OpaqueValueExpr
// BOUNDS-CHECK: |     | | |       | | |       `-DeclRefExpr {{.*}} 'x' 'int'
// BOUNDS-CHECK: |     |-MaterializeSequenceExpr {{.*}} <Unbind>
// BOUNDS-CHECK: |     | |-BinaryOperator {{.*}} 'int' lvalue '='
// BOUNDS-CHECK: |     | | |-DeclRefExpr {{.*}} 'y'
// BOUNDS-CHECK: |     | | `-OpaqueValueExpr
// BOUNDS-CHECK: |     | |     `-DeclRefExpr {{.*}} 'x'
// BOUNDS-CHECK: |     | |-OpaqueValueExpr
// BOUNDS-CHECK: |     | |   `-MaterializeSequenceExpr {{.*}} <Unbind>
// BOUNDS-CHECK: |     | |     |-MaterializeSequenceExpr {{.*}} <Bind>
// BOUNDS-CHECK: |     | |     | |-BoundsSafetyPointerPromotionExpr {{.*}} 'int *__bidi_indexable'
// BOUNDS-CHECK: |     | |     | | |-OpaqueValueExpr {{.*}}
// BOUNDS-CHECK: |     | |     | | | `-CallExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |     | |     | | |   `-OpaqueValueExpr
// BOUNDS-CHECK: |     | |     | | |       `-DeclRefExpr {{.*}} 'x' 'int'
// BOUNDS-CHECK: |     `-ReturnStmt
// BOUNDS-CHECK: |         `-UnaryOperator {{.*}} '*'
// BOUNDS-CHECK: |           `-MaterializeSequenceExpr {{.*}} <Unbind>
// BOUNDS-CHECK: |             |-MaterializeSequenceExpr {{.*}} <Bind>
// BOUNDS-CHECK: |             | |-BoundsSafetyPointerPromotionExpr {{.*}} 'int *__bidi_indexable'
// BOUNDS-CHECK: |             | | |-OpaqueValueExpr
// BOUNDS-CHECK: |             | | |   `-DeclRefExpr {{.*}} 'p' 'int *__single __sized_by_or_null(4UL * y)':'int *__single'
// BOUNDS-CHECK: |             | | | `-BinaryOperator {{.*}} 'char *' '+'
// BOUNDS-CHECK: |             | | |   |-CStyleCastExpr {{.*}} 'char *' <BitCast>
// BOUNDS-CHECK: |             | | |   | `-ImplicitCastExpr {{.*}} 'int *' <BoundsSafetyPointerCast> part_of_explicit_cast
// BOUNDS-CHECK: |             | | |   |   `-OpaqueValueExpr
// BOUNDS-CHECK: |             | | |   |       `-DeclRefExpr {{.*}} 'p' 'int *__single __sized_by_or_null(4UL * y)':'int *__single'
// BOUNDS-CHECK: |             | | |   `-AssumptionExpr
// BOUNDS-CHECK: |             | | |     `-BinaryOperator {{.*}} 'bool' '>='
// BOUNDS-CHECK: |             | | |       | `-BinaryOperator {{.*}} 'unsigned long' '*'
// BOUNDS-CHECK: |             | | |       |   |-IntegerLiteral {{.*}} 4
// BOUNDS-CHECK: |             | | |       |   `-DeclRefExpr {{.*}} 'y'
// BOUNDS-CHECK: |             | | |       `-IntegerLiteral {{.*}} 0

void foo(int m) {
    int * p1 = mymalloc<int>(m);
    int * p2 = mymalloc<int>(10);
    int i = bar<int>(m);
}

// ATTR-ONLY: |-FunctionDecl {{.*}} foo
// ATTR-ONLY: |     `-VarDecl {{.*}} p1 'int *' cinit
// ATTR-ONLY: |       `-CallExpr {{.*}} 'int * __sized_by_or_null(4UL * n)':'int *'
// ATTR-ONLY: |           `-DeclRefExpr {{.*}} 'm' 'int'
// ATTR-ONLY: |     `-VarDecl {{.*}} p2 'int *' cinit
// ATTR-ONLY: |       `-CallExpr {{.*}} 'int * __sized_by_or_null(4UL * n)':'int *'
// ATTR-ONLY: |         `-IntegerLiteral {{.*}} 'int' 10

// BOUNDS-CHECK: |-FunctionDecl {{.*}} foo
// BOUNDS-CHECK: |    `-VarDecl {{.*}} p1 'int *__bidi_indexable' cinit
// BOUNDS-CHECK: |      `-MaterializeSequenceExpr {{.*}} 'int *__bidi_indexable' <Unbind>
// BOUNDS-CHECK: |        |-MaterializeSequenceExpr {{.*}} 'int *__bidi_indexable' <Bind>
// BOUNDS-CHECK: |        | |-BoundsSafetyPointerPromotionExpr {{.*}} 'int *__bidi_indexable'
// BOUNDS-CHECK: |        | | |-OpaqueValueExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |        | | | `-CallExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |        | | |   `-OpaqueValueExpr {{.*}} 'int'
// BOUNDS-CHECK: |        | | |     `-DeclRefExpr {{.*}} 'm' 'int'
// BOUNDS-CHECK: |     `-VarDecl {{.*}} p2 'int *__bidi_indexable' cinit
// BOUNDS-CHECK: |       `-MaterializeSequenceExpr {{.*}} 'int *__bidi_indexable' <Unbind>
// BOUNDS-CHECK: |         |-MaterializeSequenceExpr {{.*}} 'int *__bidi_indexable' <Bind>
// BOUNDS-CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.*}} 'int *__bidi_indexable'
// BOUNDS-CHECK: |         | | |-OpaqueValueExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |         | | | `-CallExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |         | | |   `-OpaqueValueExpr {{.*}} 'int'
// BOUNDS-CHECK: |         | | |     `-IntegerLiteral {{.*}} 'int' 10

template <typename T>
struct Outer {
    struct Inner {
        T size;
        T * __sized_by_or_null(size) p_m;
        T * __sized_by_or_null(sizeof(T) * n) mymalloc_m(int n) {
            return static_cast<T *>(malloc(sizeof(T) * n));
        }

        void bar_m(int q) {
            this->p_m = mymalloc_m(q);
            this->size = q;
        }
    } inner;

    void foo_m(int m) {
        T l;
        T * __sized_by_or_null(l) p1;
        p1 = inner.mymalloc_m(m);
        l = m;
    }
};

void method_call(int o) {
    struct Outer<int> outer;
    outer.foo_m(o);
    outer.inner.bar_m(o);
    int r  = outer.inner.size;
    int * __sized_by_or_null(r) p1 = outer.inner.p_m;
}

// ATTR-ONLY: |-ClassTemplateDecl {{.*}} Outer
// ATTR-ONLY: | |-TemplateTypeParmDecl {{.*}} T
// ATTR-ONLY: | |-CXXRecordDecl {{.*}} struct Outer
// ATTR-ONLY: | | |-CXXRecordDecl {{.*}} struct Inner
// ATTR-ONLY: | | | |-FieldDecl {{.*}} size 'T'
// ATTR-ONLY: | | | |-FieldDecl {{.*}} p_m 'T *'
// ATTR-ONLY: | | | | `-SizedByOrNullAttr
// ATTR-ONLY: | | | |   `-MemberExpr {{.*}} 'T' lvalue ->size
// ATTR-ONLY: | | | |     `-CXXThisExpr {{.*}} 'Outer::Inner *'
// ATTR-ONLY: | | | |-CXXMethodDecl {{.*}} mymalloc_m 'T *(int)'
// ATTR-ONLY: | | | | |-CompoundStmt
// ATTR-ONLY: | | | | | `-ReturnStmt
// ATTR-ONLY: | | | | |   `-CXXStaticCastExpr {{.*}} 'T *' static_cast<T *> <Dependent>
// ATTR-ONLY: | | | | |     `-CallExpr {{.*}} 'void * __sized_by_or_null(size)':'void *'
// ATTR-ONLY: | | | | |         `-BinaryOperator {{.*}} '*'
// ATTR-ONLY: | | | | |           |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// ATTR-ONLY: | | | | |           `-DeclRefExpr {{.*}} 'n' 'int'
// ATTR-ONLY: | | | | `-SizedByOrNullAttr
// ATTR-ONLY: | | | |   `-BinaryOperator {{.*}} '*'
// ATTR-ONLY: | | | |     |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// ATTR-ONLY: | | | |     `-DeclRefExpr {{.*}} 'n' 'int'
// ATTR-ONLY: | | | `-CXXMethodDecl {{.*}} bar_m 'void (int)'
// ATTR-ONLY: | | |   `-CompoundStmt
// ATTR-ONLY: | | |     |-BinaryOperator {{.*}} '='
// ATTR-ONLY: | | |     | |-MemberExpr {{.*}} ->p_m
// ATTR-ONLY: | | |     | | `-CXXThisExpr {{.*}} 'Outer::Inner *' this
// ATTR-ONLY: | | |     | `-CallExpr {{.*}}
// ATTR-ONLY: | | |     |   |-MemberExpr {{.*}} '<bound member function type>' ->mymalloc_m
// ATTR-ONLY: | | |     |   | `-CXXThisExpr {{.*}} 'Outer::Inner *'
// ATTR-ONLY: | | |     |   `-DeclRefExpr {{.*}} 'q' 'int'
// ATTR-ONLY: | | |     `-BinaryOperator {{.*}} '<dependent type>' '='
// ATTR-ONLY: | | |       |-MemberExpr {{.*}} 'T' lvalue ->size
// ATTR-ONLY: | | |       | `-CXXThisExpr {{.*}} 'Outer::Inner *'
// ATTR-ONLY: | | |       `-DeclRefExpr {{.*}} 'q' 'int'
// ATTR-ONLY: | | |-FieldDecl {{.*}} inner 'struct Inner':'Outer::Inner'
// ATTR-ONLY: | | `-CXXMethodDecl {{.*}} foo_m 'void (int)'
// ATTR-ONLY: | |   `-CompoundStmt {{.*}}
// ATTR-ONLY: | |     | `-VarDecl {{.*}} l 'T'
// ATTR-ONLY: | |     | `-VarDecl {{.*}} p1 'T *'
// ATTR-ONLY: | |     |   `-SizedByOrNullAttr
// ATTR-ONLY: | |     |     `-DeclRefExpr {{.*}} 'l' 'T'
// ATTR-ONLY: | |     |-BinaryOperator {{.*}} '<dependent type>' '='
// ATTR-ONLY: | |     | |-DeclRefExpr {{.*}} 'p1' 'T *'
// ATTR-ONLY: | |     | `-CallExpr {{.*}} '<dependent type>'
// ATTR-ONLY: | |     |   |-CXXDependentScopeMemberExpr {{.*}} .mymalloc_m
// ATTR-ONLY: | |     |   | `-MemberExpr {{.*}} 'struct Inner':'Outer::Inner' lvalue ->inner
// ATTR-ONLY: | |     |   | `-CXXThisExpr {{.*}} 'Outer<T> *'
// ATTR-ONLY: | |     |   `-DeclRefExpr {{.*}} 'm' 'int'
// ATTR-ONLY: | |     `-BinaryOperator {{.*}} '='
// ATTR-ONLY: | |       |-DeclRefExpr {{.*}} 'l' 'T'
// ATTR-ONLY: | |       `-DeclRefExpr {{.*}} 'm' 'int'

// BOUNDS-CHECK: |-ClassTemplateDecl {{.*}} Outer
// BOUNDS-CHECK: | |-TemplateTypeParmDecl {{.*}} T
// BOUNDS-CHECK: | |-CXXRecordDecl {{.*}} struct Outer
// BOUNDS-CHECK: | | |-CXXRecordDecl {{.*}} struct Inner
// BOUNDS-CHECK: | | | |-FieldDecl {{.*}} size 'T'
// BOUNDS-CHECK: | | | |-FieldDecl {{.*}} p_m 'T *__single'
// BOUNDS-CHECK: | | | | `-SizedByOrNullAttr
// BOUNDS-CHECK: | | | |   `-MemberExpr {{.*}} 'T' lvalue ->size
// BOUNDS-CHECK: | | | |     `-CXXThisExpr {{.*}} 'Outer::Inner *'
// BOUNDS-CHECK: | | | |-CXXMethodDecl {{.*}} mymalloc_m 'T *__single(int)'
// BOUNDS-CHECK: | | | | |-ParmVarDecl {{.*}} n 'int'
// BOUNDS-CHECK: | | | | |-CompoundStmt
// BOUNDS-CHECK: | | | | | `-ReturnStmt
// BOUNDS-CHECK: | | | | |   `-CXXStaticCastExpr {{.*}} 'T *' static_cast<T *> <Dependent>
// BOUNDS-CHECK: | | | | |     `-CallExpr {{.*}} 'void *__single __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: | | | | |       | `-DeclRefExpr {{.*}} 'malloc'
// BOUNDS-CHECK: | | | | |       `-BinaryOperator {{.*}} '*'
// BOUNDS-CHECK: | | | | |         |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// BOUNDS-CHECK: | | | | |         `-DeclRefExpr {{.*}} 'n' 'int'
// BOUNDS-CHECK: | | | | `-SizedByOrNullAttr
// BOUNDS-CHECK: | | | |   `-BinaryOperator {{.*}} '*'
// BOUNDS-CHECK: | | | |     |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'T'
// BOUNDS-CHECK: | | | |     `-DeclRefExpr {{.*}} 'n'
// BOUNDS-CHECK: | | | `-CXXMethodDecl {{.*}}  bar_m 'void (int)'
// BOUNDS-CHECK: | | |   `-CompoundStmt
// BOUNDS-CHECK: | | |     |-BinaryOperator {{.*}} '='
// BOUNDS-CHECK: | | |     | |-MemberExpr {{.*}} 'T *__single' lvalue ->p_m
// BOUNDS-CHECK: | | |     | | `-CXXThisExpr {{.*}} 'Outer::Inner *'
// BOUNDS-CHECK: | | |     | `-CallExpr {{.*}} '<dependent type>'
// BOUNDS-CHECK: | | |     |   |-MemberExpr {{.*}} '<bound member function type>' ->mymalloc_m
// BOUNDS-CHECK: | | |     |   | `-CXXThisExpr {{.*}} 'Outer::Inner *'
// BOUNDS-CHECK: | | |     |   `-DeclRefExpr {{.*}} 'q'
// BOUNDS-CHECK: | | |     `-BinaryOperator {{.*}} '='
// BOUNDS-CHECK: | | |       |-MemberExpr {{.*}} 'T' lvalue ->size
// BOUNDS-CHECK: | | |       | `-CXXThisExpr {{.*}} 'Outer::Inner *'
// BOUNDS-CHECK: | | |       `-DeclRefExpr {{.*}} 'q'
// BOUNDS-CHECK: | | |-FieldDecl {{.*}} inner 'struct Inner':'Outer::Inner'
// BOUNDS-CHECK: | | `-CXXMethodDecl {{.*}} foo_m 'void (int)'
// BOUNDS-CHECK: | |   `-CompoundStmt
// BOUNDS-CHECK: | |     | `-VarDecl {{.*}} l 'T'
// BOUNDS-CHECK: | |     | `-VarDecl {{.*}} p1 'T *__bidi_indexable'
// BOUNDS-CHECK: | |     |   `-SizedByOrNullAttr
// BOUNDS-CHECK: | |     |     `-DeclRefExpr {{.*}} 'l' 'T'
// BOUNDS-CHECK: | |     |-BinaryOperator {{.*}} '<dependent type>' '='
// BOUNDS-CHECK: | |     | |-DeclRefExpr {{.*}} 'p1' 'T *__bidi_indexable'
// BOUNDS-CHECK: | |     | `-CallExpr {{.*}} '<dependent type>'
// BOUNDS-CHECK: | |     |   |-CXXDependentScopeMemberExpr {{.*}} .mymalloc_m
// BOUNDS-CHECK: | |     |   | `-MemberExpr {{.*}} 'struct Inner':'Outer::Inner' lvalue ->inner
// BOUNDS-CHECK: | |     |   | `-CXXThisExpr {{.*}} 'Outer<T> *'
// BOUNDS-CHECK: | |     |   `-DeclRefExpr {{.*}} 'm' 'int'
// BOUNDS-CHECK: | |     `-BinaryOperator {{.*}} '='
// BOUNDS-CHECK: | |       |-DeclRefExpr {{.*}} 'l' 'T'
// BOUNDS-CHECK: | |       `-DeclRefExpr {{.*}} 'm' 'int'

// ATTR-ONLY: | `-ClassTemplateSpecializationDecl {{.*}} struct Outer
// ATTR-ONLY: |   |-TemplateArgument type 'int'
// ATTR-ONLY: |   |-CXXRecordDecl {{.*}} struct Inner
// ATTR-ONLY: |   | |-FieldDecl {{.*}} size 'int'
// ATTR-ONLY: |   | | `-DependerDeclsAttr
// ATTR-ONLY: |   | |-FieldDecl {{.*}} p_m 'int * __sized_by_or_null(size)':'int *'
// ATTR-ONLY: |   | |-CXXMethodDecl {{.*}} mymalloc_m 'int * __sized_by_or_null(4UL * n)(int)'
// ATTR-ONLY: |   | | `-CompoundStmt
// ATTR-ONLY: |   | |   `-ReturnStmt
// ATTR-ONLY: |   | |     `-CXXStaticCastExpr {{.*}} static_cast<int *>
// ATTR-ONLY: |   | |       `-CallExpr
// ATTR-ONLY: |   | |         | `-DeclRefExpr {{.*}} 'malloc' 'void * __sized_by_or_null(size)(int)'
// ATTR-ONLY: |   | |         `-BinaryOperator {{.*}} '*'
// ATTR-ONLY: |   | |           |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'int'
// ATTR-ONLY: |   | |           `-DeclRefExpr {{.*}} 'n' 'int'

// BOUNDS-CHECK: | `-ClassTemplateSpecializationDecl {{.*}}  struct Outer
// BOUNDS-CHECK: |   |-TemplateArgument type 'int'
// BOUNDS-CHECK: |   |-CXXRecordDecl {{.*}} struct Outer
// BOUNDS-CHECK: |   |-CXXRecordDecl {{.*}} struct Inner
// BOUNDS-CHECK: |   | |-FieldDecl {{.*}} size 'int'
// BOUNDS-CHECK: |   | | `-DependerDeclsAttr {{.*}} 0
// BOUNDS-CHECK: |   | |-FieldDecl {{.*}} p_m 'int *__single __sized_by_or_null(size)':'int *__single'
// BOUNDS-CHECK: |   | |-CXXMethodDecl {{.*}} mymalloc_m 'int *__single __sized_by_or_null(4UL * n)(int)'
// BOUNDS-CHECK: |   | | `-CompoundStmt
// BOUNDS-CHECK: |   | |   `-ReturnStmt
// BOUNDS-CHECK: |   | |     `-CXXStaticCastExpr {{.*}} 'int *' static_cast<int *> <BitCast>
// BOUNDS-CHECK: |   | |       `-MaterializeSequenceExpr {{.*}} 'void *__bidi_indexable' <Unbind>
// BOUNDS-CHECK: |   | |         |-MaterializeSequenceExpr {{.*}} 'void *__bidi_indexable' <Bind>
// BOUNDS-CHECK: |   | |         | |-BoundsSafetyPointerPromotionExpr {{.*}} 'void *__bidi_indexable'
// BOUNDS-CHECK: |   | |         | | |-OpaqueValueExpr {{.*}} 'void *__single __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: |   | |         | | | `-CallExpr {{.*}} 'void *__single __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: |   | |         | | |   | `-DeclRefExpr {{.*}} 'malloc' 'void *__single __sized_by_or_null(size)(int)'
// BOUNDS-CHECK: |   | |         | | |   `-OpaqueValueExpr {{.*}} 'int'
// BOUNDS-CHECK: |   | |         | | |     `-BinaryOperator {{.*}} '*'
// BOUNDS-CHECK: |   | |         | | |       |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'int'
// BOUNDS-CHECK: |   | |         | | |       `-DeclRefExpr {{.*}} 'n'
// BOUNDS-CHECK: |   | |         | | | `-BinaryOperator {{.*}} 'char *' '+'
// BOUNDS-CHECK: |   | |         | | |   |-CStyleCastExpr {{.*}} 'char *' <BitCast>
// BOUNDS-CHECK: |   | |         | | |   | `-OpaqueValueExpr {{.*}} 'void *__single __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: |   | |         | | |   |   `-CallExpr {{.*}} 'void *__single __sized_by_or_null(size)':'void *__single'
// BOUNDS-CHECK: |   | |         | | |   |     | `-DeclRefExpr {{.*}} 'malloc'
// BOUNDS-CHECK: |   | |         | | |   `-OpaqueValueExpr {{.*}} 'int'
// BOUNDS-CHECK: |   | |         | | |     `-BinaryOperator {{.*}} '*'
// BOUNDS-CHECK: |   | |         | | |       |-UnaryExprOrTypeTraitExpr {{.*}} sizeof 'int'
// BOUNDS-CHECK: |   | |         | | |       `-DeclRefExpr {{.*}} 'n'
// BOUNDS-CHECK: |   | |         | | `-<<<NULL>>>

// ATTR-ONLY: |   | |-CXXMethodDecl {{.*}} bar_m 'void (int)'
// ATTR-ONLY: |   | | `-CompoundStmt {{.*}}
// ATTR-ONLY: |   | |   |-BinaryOperator {{.*}} 'int * __sized_by_or_null(size)':'int *' lvalue '='
// ATTR-ONLY: |   | |   | |-MemberExpr {{.*}} ->p_m
// ATTR-ONLY: |   | |   | | `-CXXThisExpr {{.*}} 'Outer<int>::Inner *'
// ATTR-ONLY: |   | |   | `-CXXMemberCallExpr {{.*}} 'int * __sized_by_or_null(4UL * n)':'int *'
// ATTR-ONLY: |   | |   |   |-MemberExpr {{.*}} '<bound member function type>' ->mymalloc_m
// ATTR-ONLY: |   | |   |   | `-CXXThisExpr {{.*}} 'Outer<int>::Inner *'
// ATTR-ONLY: |   | |   |   `-ImplicitCastExpr {{.*}}
// ATTR-ONLY: |   | |   |     `-DeclRefExpr {{.*}} 'q' 'int'
// ATTR-ONLY: |   | |   `-BinaryOperator {{.*}} 'int' lvalue '='
// ATTR-ONLY: |   | |     |-MemberExpr {{.*}} ->size
// ATTR-ONLY: |   | |     | `-CXXThisExpr {{.*}} 'Outer<int>::Inner *'
// ATTR-ONLY: |   | |     `-ImplicitCastExpr {{.*}}
// ATTR-ONLY: |   | |       `-DeclRefExpr {{.*}} 'q' 'int'

// BOUNDS-CHECK: |   | |-CXXMethodDecl {{.*}} bar_m 'void (int)'
// BOUNDS-CHECK: |   | | |-ParmVarDecl {{.*}} q 'int'
// BOUNDS-CHECK: |   | | `-CompoundStmt
// BOUNDS-CHECK: |   | |   |-MaterializeSequenceExpr
// BOUNDS-CHECK: |   | |   | |-BoundsCheckExpr {{.*}} 'this->mymalloc_m(q) <= __builtin_get_pointer_upper_bound(this->mymalloc_m(q)) && __builtin_get_pointer_lower_bound(this->mymalloc_m(q)) <= this->mymalloc_m(q) && !this->mymalloc_m(q) || q <= (char *)__builtin_get_pointer_upper_bound(this->mymalloc_m(q)) - (char *__single)this->mymalloc_m(q) && 0 <= q'
// BOUNDS-CHECK: |   | |   | | |-BinaryOperator {{.*}} '='
// BOUNDS-CHECK: |   | |   | | | |-MemberExpr {{.*}} ->p_m
// BOUNDS-CHECK: |   | |   | | | | `-CXXThisExpr {{.*}} 'Outer<int>::Inner *'
// BOUNDS-CHECK: |   | |   | | | `-OpaqueValueExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |   | |   | | |   `-CXXMemberCallExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |   | |   | | |     |-MemberExpr {{.*}} '<bound member function type>' ->mymalloc_m
// BOUNDS-CHECK: |   | |   | | |     | `-CXXThisExpr {{.*}} 'Outer<int>::Inner *'
// BOUNDS-CHECK: |   | |   | | |     `-DeclRefExpr {{.*}} 'q'
// BOUNDS-CHECK: |   | |   `-MaterializeSequenceExpr {{.*}} 'int' lvalue <Unbind>
// BOUNDS-CHECK: |   | |     |-BinaryOperator {{.*}} 'int' lvalue '='
// BOUNDS-CHECK: |   | |     | |-MemberExpr {{.*}} 'int' lvalue ->size
// BOUNDS-CHECK: |   | |     | | `-CXXThisExpr {{.*}} 'Outer<int>::Inner *'
// BOUNDS-CHECK: |   | |     | `-OpaqueValueExpr {{.*}} 'int'
// BOUNDS-CHECK: |   | |     |   `-DeclRefExpr {{.*}} 'q' 'int'

// ATTR-ONLY: |   |-FieldDecl {{.*}} 'struct Inner':'Outer<int>::Inner'

// BOUNDS-CHECK: |   |-FieldDecl {{.*}} inner 'struct Inner':'Outer<int>::Inner'

// ATTR-ONLY: |   |-CXXMethodDecl {{.*}} foo_m 'void (int)'
// ATTR-ONLY: |   | `-CompoundStmt {{.*}}
// ATTR-ONLY: |   |   | `-VarDecl {{.*}} l 'int'
// ATTR-ONLY: |   |   |   `-DependerDeclsAttr
// ATTR-ONLY: |   |   | `-VarDecl {{.*}} p1 'int * __sized_by_or_null(l)':'int *'
// ATTR-ONLY: |   |   |-BinaryOperator {{.*}} '='
// ATTR-ONLY: |   |   | |-DeclRefExpr {{.*}} 'p1' 'int * __sized_by_or_null(l)':'int *'
// ATTR-ONLY: |   |   | `-CXXMemberCallExpr {{.*}} 'int * __sized_by_or_null(4UL * n)':'int *'
// ATTR-ONLY: |   |   |   |-MemberExpr {{.*}} '<bound member function type>' .mymalloc_m
// ATTR-ONLY: |   |   |   | `-MemberExpr {{.*}} ->inner
// ATTR-ONLY: |   |   |   | `-CXXThisExpr {{.*}} 'Outer<int> *'
// ATTR-ONLY: |   |   |   `-DeclRefExpr {{.*}} 'm' 'int'
// ATTR-ONLY: |   |   `-BinaryOperator {{.*}} '='
// ATTR-ONLY: |   |     |-DeclRefExpr {{.*}} 'l' 'int'
// ATTR-ONLY: |   |     `-DeclRefExpr {{.*}} 'm' 'int'

// BOUNDS-CHECK: |   |-CXXMethodDecl {{.*}} foo_m 'void (int)'
// BOUNDS-CHECK: |   | `-CompoundStmt {{.*}}
// BOUNDS-CHECK: |   |   | `-VarDecl {{.*}} l 'int'
// BOUNDS-CHECK: |   |   |   `-DependerDeclsAttr
// BOUNDS-CHECK: |   |   | `-VarDecl {{.*}} p1 'int *__single __sized_by_or_null(l)':'int *__single'
// BOUNDS-CHECK: |   |   |-MaterializeSequenceExpr {{.*}} <Bind>
// BOUNDS-CHECK: |   |   | |-BoundsCheckExpr {{.*}} 'this->inner.mymalloc_m(m) <= __builtin_get_pointer_upper_bound(this->inner.mymalloc_m(m)) && __builtin_get_pointer_lower_bound(this->inner.mymalloc_m(m)) <= this->inner.mymalloc_m(m) && !this->inner.mymalloc_m(m) || m <= (char *)__builtin_get_pointer_upper_bound(this->inner.mymalloc_m(m)) - (char *__single)this->inner.mymalloc_m(m) && 0 <= m'
// BOUNDS-CHECK: |   |   | | |-BinaryOperator {{.*}} '='
// BOUNDS-CHECK: |   |   | | | |-DeclRefExpr {{.*}} 'p1' 'int *__single __sized_by_or_null(l)':'int *__single'
// BOUNDS-CHECK: |   |   | | | `-CXXMemberCallExpr {{.*}} 'int *__single __sized_by_or_null(4UL * n)':'int *__single'
// BOUNDS-CHECK: |   |   | | |   |-MemberExpr {{.*}} '<bound member function type>' .mymalloc_m
// BOUNDS-CHECK: |   |   | | |   | `-MemberExpr {{.*}} ->inner
// BOUNDS-CHECK: |   |   | | |   | `-CXXThisExpr {{.*}} 'Outer<int> *'
// BOUNDS-CHECK: |   |   | | |   `-DeclRefExpr {{.*}} 'm' 'int'
// BOUNDS-CHECK: |   |   `-MaterializeSequenceExpr {{.*}} <Unbind>
// BOUNDS-CHECK: |   |     |-BinaryOperator {{.*}} '='
// BOUNDS-CHECK: |   |     | |-DeclRefExpr {{.*}} 'l' 'int'
// BOUNDS-CHECK: |   |     | `-DeclRefExpr {{.*}} 'm' 'int'

// rdar://144585904 ([BoundsSafety][C++] Applying bounds attribute to lambda fails)
#ifndef BOUNDS_SAFETY // crashes clang with -fbounds-safety -fbounds-attributes-cxx-experimental enabled
template <typename T>
void lambda(T h) {
    auto g = [](T i, T * __sized_by_or_null(i) p1) -> T * __sized_by_or_null(i) {
        T j = i;
        T * __sized_by_or_null(j) p2 = p1;
        return p2;
    };
    auto f = [&g](T i) {
        T j;
        T * __sized_by_or_null(j) p;
        p = mymalloc<T>(i);
        j = i;

        p = g(j, p);
        j = j;
    };
    f(h);
}

template void lambda<int>(int h);
#endif

// ATTR-ONLY: `-FunctionTemplateDecl {{.*}} lambda
// ATTR-ONLY:   |-TemplateTypeParmDecl {{.*}} T
// ATTR-ONLY:   |-FunctionDecl {{.*}} lambda 'void (T)'
// ATTR-ONLY:   | `-CompoundStmt
// ATTR-ONLY:   |   |-DeclStmt
// ATTR-ONLY:   |   | `-VarDecl {{.*}} g 'auto'
// ATTR-ONLY:   |   |   `-LambdaExpr
// ATTR-ONLY:   |   |     `-CompoundStmt {{.*}}
// ATTR-ONLY:   |   |       | `-VarDecl {{.*}} j 'T'
// ATTR-ONLY:   |   |       |   `-DeclRefExpr {{.*}} 'i' 'T'
// ATTR-ONLY:   |   |       | `-VarDecl {{.*}} p2 'T *'
// ATTR-ONLY:   |   |       |   |-DeclRefExpr {{.*}} 'p1' 'T *'
// ATTR-ONLY:   |   |       |   `-SizedByOrNullAttr
// ATTR-ONLY:   |   |       |     `-DeclRefExpr {{.*}} 'j' 'T'
// ATTR-ONLY:   |   |       `-ReturnStmt
// ATTR-ONLY:   |   |         `-DeclRefExpr {{.*}} 'p2' 'T *'
// ATTR-ONLY:   |   |-DeclStmt
// ATTR-ONLY:   |   | `-VarDecl {{.*}} f 'auto'
// ATTR-ONLY:   |   |   `-LambdaExpr
// ATTR-ONLY:   |   |     `-CompoundStmt {{.*}}
// ATTR-ONLY:   |   |       | `-VarDecl {{.*}} j 'T'
// ATTR-ONLY:   |   |       | `-VarDecl {{.*}} p 'T *'
// ATTR-ONLY:   |   |       |   `-SizedByOrNullAttr
// ATTR-ONLY:   |   |       |     `-DeclRefExpr {{.*}} 'j' 'T'
// ATTR-ONLY:   |   |       |-BinaryOperator {{.*}} '<dependent type>' '='
// ATTR-ONLY:   |   |       | |-DeclRefExpr {{.*}} 'p' 'T *'
// ATTR-ONLY:   |   |       | `-CallExpr {{.*}} '<dependent type>'
// ATTR-ONLY:   |   |       |   |-UnresolvedLookupExpr {{.*}} 'mymalloc'
// ATTR-ONLY:   |   |       |   | `-TemplateArgument type 'T'
// ATTR-ONLY:   |   |       |   `-DeclRefExpr {{.*}} 'i' 'T'
// ATTR-ONLY:   |   |       |-BinaryOperator {{.*}} '<dependent type>' '='
// ATTR-ONLY:   |   |       | |-DeclRefExpr {{.*}} 'j' 'T'
// ATTR-ONLY:   |   |       | `-DeclRefExpr {{.*}} 'i' 'T'
// ATTR-ONLY:   |   |       |-BinaryOperator {{.*}} '<dependent type>' '='
// ATTR-ONLY:   |   |       | |-DeclRefExpr {{.*}} 'p' 'T *'
// ATTR-ONLY:   |   |       | `-CallExpr {{.*}} '<dependent type>'
// ATTR-ONLY:   |   |       |   |-DeclRefExpr {{.*}} 'g' 'auto' refers_to_enclosing_variable_or_capture
// ATTR-ONLY:   |   |       |   |-DeclRefExpr {{.*}} 'j' 'T'
// ATTR-ONLY:   |   |       |   `-DeclRefExpr {{.*}} 'p' 'T *'
// ATTR-ONLY:   |   |       `-BinaryOperator {{.*}} '<dependent type>' '='
// ATTR-ONLY:   |   |         |-DeclRefExpr {{.*}} 'j' 'T'
// ATTR-ONLY:   |   |         `-DeclRefExpr {{.*}} 'j' 'T'
// ATTR-ONLY:   |   `-CallExpr {{.*}} '<dependent type>'
// ATTR-ONLY:   |     |-DeclRefExpr {{.*}} 'f' 'auto'
// ATTR-ONLY:   |     `-DeclRefExpr {{.*}} 'h' 'T'
// ATTR-ONLY:   `-FunctionDecl {{.*}} lambda 'void (int)' explicit_instantiation_definition
// ATTR-ONLY:     |-TemplateArgument type 'int'
// ATTR-ONLY:     `-CompoundStmt {{.*}}
// ATTR-ONLY:       |-DeclStmt {{.*}}
// ATTR-ONLY:       | `-VarDecl {{.*}} g
// ATTR-ONLY:       |   `-LambdaExpr
// ATTR-ONLY:       |     |-ParmVarDecl {{.*}} i 'int'
//                        FIXME: apply attribute to lambda parameter
// ATTR-ONLY:       |     |-ParmVarDecl {{.*}} p1 'int *'
// ATTR-ONLY:       |     `-CompoundStmt
// ATTR-ONLY:       |       | `-VarDecl {{.*}}j 'int'
// ATTR-ONLY:       |       |   | `-DeclRefExpr {{.*}} 'i' 'int'
// ATTR-ONLY:       |       |   `-DependerDeclsAttr
// ATTR-ONLY:       |       | `-VarDecl {{.*}} p2 'int * __sized_by_or_null(j)':'int *'
// ATTR-ONLY:       |       |   `-DeclRefExpr {{.*}} 'p1' 'int *'
// ATTR-ONLY:       |       `-ReturnStmt {{.*}}
// ATTR-ONLY:       |         `-ImplicitCastExpr {{.*}}  'int * __sized_by_or_null(j)':'int *' <LValueToRValue>
// ATTR-ONLY:       |           `-DeclRefExpr {{.*}} 'p2' 'int * __sized_by_or_null(j)':'int *'
// ATTR-ONLY:       |-DeclStmt {{.*}}
// ATTR-ONLY:       | `-VarDecl {{.*}} f
// ATTR-ONLY:       |   `-LambdaExpr
// ATTR-ONLY:       |     `-CompoundStmt
// ATTR-ONLY:       |       | `-VarDecl {{.*}} j 'int'
// ATTR-ONLY:       |       |   `-DependerDeclsAttr
// ATTR-ONLY:       |       | `-VarDecl {{.*}} p 'int * __sized_by_or_null(j)':'int *'
// ATTR-ONLY:       |       |-BinaryOperator {{.*}} 'int * __sized_by_or_null(j)':'int *' lvalue '='
// ATTR-ONLY:       |       | |-DeclRefExpr {{.*}} 'p' 'int * __sized_by_or_null(j)':'int *'
// ATTR-ONLY:       |       | `-CallExpr {{.*}} 'int * __sized_by_or_null(4UL * n)':'int *'
// ATTR-ONLY:       |       |   | `-DeclRefExpr {{.*}} 'mymalloc'
// ATTR-ONLY:       |       |   `-DeclRefExpr {{.*}} 'i' 'int'
// ATTR-ONLY:       |       |-BinaryOperator {{.*}} '='
// ATTR-ONLY:       |       | |-DeclRefExpr {{.*}} 'j' 'int'
// ATTR-ONLY:       |       | `-DeclRefExpr {{.*}} 'i' 'int'
// ATTR-ONLY:       |       |-BinaryOperator {{.*}} '='
// ATTR-ONLY:       |       | |-DeclRefExpr {{.*}} 'p' 'int * __sized_by_or_null(j)':'int *'
// ATTR-ONLY:       |       | `-CXXOperatorCallExpr {{.*}} 'int *' '()'
// ATTR-ONLY:       |       |   | `-DeclRefExpr {{.*}} 'operator()' 'auto (int, int *) const -> int *'
// ATTR-ONLY:       |       |   | `-DeclRefExpr {{.*}} 'g'
// ATTR-ONLY:       |       |   | `-DeclRefExpr {{.*}} 'j' 'int'
// ATTR-ONLY:       |       |   `-DeclRefExpr {{.*}} 'p' 'int * __sized_by_or_null(j)':'int *'
// ATTR-ONLY:       |       `-BinaryOperator {{.*}} '='
// ATTR-ONLY:       |         |-DeclRefExpr {{.*}} 'j' 'int'
// ATTR-ONLY:       |         `-DeclRefExpr {{.*}} 'j' 'int'
// ATTR-ONLY:       `-CXXOperatorCallExpr {{.*}}  'void' '()'
// ATTR-ONLY:         | `-DeclRefExpr {{.*}} 'operator()' 'void (int) const'
// ATTR-ONLY:         | `-DeclRefExpr {{.*}} 'f'
// ATTR-ONLY:           `-DeclRefExpr {{.*}} 'h' 'int'
