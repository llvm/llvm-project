// RUN: %clang_cc1 -fexperimental-bounds-safety-attributes -ast-dump %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

template <class T>
class RefParamMustBePtrGood {
    public:
    T __single ptr0;
    T __unsafe_indexable ptr1;

    T __single get_single() const {
        T tmp = ptr0;
        return tmp;
    }

    T __unsafe_indexable get_unsafe_indexable() const {
        T tmp = ptr1;
        return tmp;
    }

    void set_single(T __single v) {
        ptr0 = v;
    }
    
    void set_single() {
        T __single tmp = nullptr;
        ptr0 = tmp;
    }

    void set_unsafe_indexable() {
        T __unsafe_indexable tmp = nullptr;
        ptr1 = tmp;
    }

    void set_unsafe_indexable(T __unsafe_indexable v) {
        ptr0 = v;
    }
};

// Do not instantiate this class
template <class T>
class RefParamMustBePtrNeverInstantiated {
    public:
    T __single ptr0;
    T __unsafe_indexable ptr1;

    T __single get_single() const {
        T tmp = ptr0;
        return tmp;
    }

    T __unsafe_indexable get_unsafe_indexable() const {
        T tmp = ptr1;
        return tmp;
    }

    void set_single(T __single v) {
        ptr0 = v;
    }
    
    void set_single() {
        T __single tmp = nullptr;
        ptr0 = tmp;
    }

    void set_unsafe_indexable() {
        T __unsafe_indexable tmp = nullptr;
        ptr1 = tmp;
    }

    void set_unsafe_indexable(T __unsafe_indexable v) {
        ptr0 = v;
    }
};

using PtrTypedef = char*;

// Explicit instantiation with good type
template
class RefParamMustBePtrGood<float*>;

void Instantiate_RefParamMustBePtrGood() {
    // Implicit instantiation
    RefParamMustBePtrGood<int*> good0;
    RefParamMustBePtrGood<PtrTypedef> good1;
}


template <class T>
class RefParamIsPointee {
    public:
    T* __single ptr0;
    T* __unsafe_indexable pt1;
};

// Explicit instantiation
template class RefParamIsPointee<float>;

void Instantiate_RefParamIsPointee() {
    // Implicit instantiation
    RefParamIsPointee<int> good0;
    RefParamIsPointee<PtrTypedef> good1;
}

// =============================================================================
// T in method body
// =============================================================================
template <class T>
class TInMethodBodyGood {
    public:
    void test() {
        T __single tmp;
        tmp = nullptr;
    }
};

// Explicit instantiation
template class TInMethodBodyGood<float*>;

void Instantiate_TInMethodBodyGood() {
    // Implicit instantiation
    TInMethodBodyGood<int*> good0;
    good0.test();
}


// =============================================================================
// Partial specialization
// =============================================================================


template <class T, class U, class V>
class RefParamMustBePtrGoodPartialBase {
    public:
    T __single ptr0;
    U __unsafe_indexable ptr1;
    V counter;

    T __single get_ptr0() const { return ptr0; }
    U __unsafe_indexable get_ptr1() const { return ptr1; }
    V get_counter() const { return counter; }

    void useT() const {
        T __single tmp = ptr0;
    }

    void useU() const {
        T __unsafe_indexable tmp = ptr1;
    }
};

// good
template <class T>
class RefParamMustBePtrGoodPartialT : public RefParamMustBePtrGoodPartialBase<T, int*, int> {
    public:
    T __single ptr2;
    typeof(RefParamMustBePtrGoodPartialBase<T, int*, int>::ptr0) ptr3;
    T __single another_method() { return ptr2; }
};

template <class U>
class RefParamMustBePtrGoodPartialU : public RefParamMustBePtrGoodPartialBase<int*, U, int> {
    public:
    U __unsafe_indexable ptr2;
    typeof(RefParamMustBePtrGoodPartialBase<int*, U, int>::ptr1) ptr3;
    U __unsafe_indexable another_method() { return ptr2; }
};

// This partial specialization is never instantiated so it doesn't produce errors.
template <class V>
class RefParamMustBePtrGoodPartialV : public RefParamMustBePtrGoodPartialBase<int, int, V> {
    public:
    V __single ptr2;
    typeof(RefParamMustBePtrGoodPartialBase<int, int, V>::ptr0) ptr3;
    V __single another_method() { return ptr2; }
};

// Explicit instantiation
template
class RefParamMustBePtrGoodPartialT<float*>;
template
class RefParamMustBePtrGoodPartialU<float*>;

void Instantiate_RefParamMustBePtrGoodPartial() {
    // Implicit instantiation
    RefParamMustBePtrGoodPartialT<int*> good0;
    RefParamMustBePtrGoodPartialU<int*> good1;
}

// external counted attributes

// Note: RefParamMustBePtrExternallyCountedGood and
// RefParamMustBePtrExternallyCountedBad should have identical type shape. They
// only have different names for the purposes of diagnostic reporting.
template <class T>
class RefParamMustBePtrExternallyCountedGood {
    public:
    int size;
    T end_ptr;
    T __counted_by(size) cb;
    T __counted_by_or_null(size) cbon;
    T __sized_by(size) sb;
    T __sized_by_or_null(size) sbon;
    T __ended_by(end_ptr) eb;

    T __counted_by(size) ret_cb() {
        return cb;
    }

    // FIXME: This produces an error diagnostic before instantiating templates
    // but it shouldn't.
    // rdar://152538978.
    // void cb_param(T __counted_by(size) ptr, int size) {}

    void useT() {
        int size_local = size;
        T __counted_by(size_local) tmp = cb;
    }
};

// Explicit instantiation
template class RefParamMustBePtrExternallyCountedGood<float*>;

void Instantiate_RefParamMustBePtrExternallyCountedGood() {
    // Implicit instantiation
    RefParamMustBePtrExternallyCountedGood<int*> good0;
    RefParamMustBePtrExternallyCountedGood<PtrTypedef> good1;
}

// Generated using `clang/utils/make-ast-dump-check.sh`

// CHECK:|-ClassTemplateDecl {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:4:1, line:37:1> line:5:7 RefParamMustBePtrGood
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:4:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:5:1, line:37:1> line:5:7 class RefParamMustBePtrGood definition
// CHECK-NEXT:| | |-DefinitionData aggregate standard_layout trivially_copyable trivial
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGood
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:6:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:7:5, col:16> col:16 referenced ptr0 'T__single':'T'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:8:5, col:26> col:26 referenced ptr1 'T__unsafe_indexable':'T'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:10:5, line:13:5> line:10:16 get_single 'T () const__single' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:35, line:13:5>
// CHECK-NEXT:| | |   |-DeclStmt {{.*}} <line:11:9, col:21>
// CHECK-NEXT:| | |   | `-VarDecl {{.*}} <col:9, col:17> col:11 referenced tmp 'T' nrvo cinit
// CHECK-NEXT:| | |   |   `-MemberExpr {{.*}} <col:17> 'T const__single':'const T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| | |   |     `-CXXThisExpr {{.*}} <col:17> 'const RefParamMustBePtrGood<T> *' implicit this
// CHECK-NEXT:| | |   `-ReturnStmt {{.*}} <line:12:9, col:16> nrvo_candidate(Var {{.*}} 'tmp' 'T')
// CHECK-NEXT:| | |     `-DeclRefExpr {{.*}} <col:16> 'T' lvalue Var {{.*}} 'tmp' 'T'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:15:5, line:18:5> line:15:26 get_unsafe_indexable 'T () const__unsafe_indexable' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:55, line:18:5>
// CHECK-NEXT:| | |   |-DeclStmt {{.*}} <line:16:9, col:21>
// CHECK-NEXT:| | |   | `-VarDecl {{.*}} <col:9, col:17> col:11 referenced tmp 'T' nrvo cinit
// CHECK-NEXT:| | |   |   `-MemberExpr {{.*}} <col:17> 'T const__unsafe_indexable':'const T' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:| | |   |     `-CXXThisExpr {{.*}} <col:17> 'const RefParamMustBePtrGood<T> *' implicit this
// CHECK-NEXT:| | |   `-ReturnStmt {{.*}} <line:17:9, col:16> nrvo_candidate(Var {{.*}} 'tmp' 'T')
// CHECK-NEXT:| | |     `-DeclRefExpr {{.*}} <col:16> 'T' lvalue Var {{.*}} 'tmp' 'T'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:20:5, line:22:5> line:20:10 set_single 'void (T__single)' implicit-inline
// CHECK-NEXT:| | | |-ParmVarDecl {{.*}} <col:21, col:32> col:32 referenced v 'T__single':'T'
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:35, line:22:5>
// CHECK-NEXT:| | |   `-BinaryOperator {{.*}} <line:21:9, col:16> '<dependent type>' '='
// CHECK-NEXT:| | |     |-MemberExpr {{.*}} <col:9> 'T__single':'T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| | |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<T> *' implicit this
// CHECK-NEXT:| | |     `-DeclRefExpr {{.*}} <col:16> 'T__single':'T' lvalue ParmVar {{.*}} 'v' 'T__single':'T'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:24:5, line:27:5> line:24:10 set_single 'void ()' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:23, line:27:5>
// CHECK-NEXT:| | |   |-DeclStmt {{.*}} <line:25:9, col:33>
// CHECK-NEXT:| | |   | `-VarDecl {{.*}} <col:9, col:26> col:20 referenced tmp 'T__single':'T' cinit
// CHECK-NEXT:| | |   |   `-CXXNullPtrLiteralExpr {{.*}} <col:26> 'std::nullptr_t'
// CHECK-NEXT:| | |   `-BinaryOperator {{.*}} <line:26:9, col:16> '<dependent type>' '='
// CHECK-NEXT:| | |     |-MemberExpr {{.*}} <col:9> 'T__single':'T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| | |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<T> *' implicit this
// CHECK-NEXT:| | |     `-DeclRefExpr {{.*}} <col:16> 'T__single':'T' lvalue Var {{.*}} 'tmp' 'T__single':'T'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:29:5, line:32:5> line:29:10 set_unsafe_indexable 'void ()' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:33, line:32:5>
// CHECK-NEXT:| | |   |-DeclStmt {{.*}} <line:30:9, col:43>
// CHECK-NEXT:| | |   | `-VarDecl {{.*}} <col:9, col:36> col:30 referenced tmp 'T__unsafe_indexable':'T' cinit
// CHECK-NEXT:| | |   |   `-CXXNullPtrLiteralExpr {{.*}} <col:36> 'std::nullptr_t'
// CHECK-NEXT:| | |   `-BinaryOperator {{.*}} <line:31:9, col:16> '<dependent type>' '='
// CHECK-NEXT:| | |     |-MemberExpr {{.*}} <col:9> 'T__unsafe_indexable':'T' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:| | |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<T> *' implicit this
// CHECK-NEXT:| | |     `-DeclRefExpr {{.*}} <col:16> 'T__unsafe_indexable':'T' lvalue Var {{.*}} 'tmp' 'T__unsafe_indexable':'T'
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:34:5, line:36:5> line:34:10 set_unsafe_indexable 'void (T__unsafe_indexable)' implicit-inline
// CHECK-NEXT:| |   |-ParmVarDecl {{.*}} <col:31, col:52> col:52 referenced v 'T__unsafe_indexable':'T'
// CHECK-NEXT:| |   `-CompoundStmt {{.*}} <col:55, line:36:5>
// CHECK-NEXT:| |     `-BinaryOperator {{.*}} <line:35:9, col:16> '<dependent type>' '='
// CHECK-NEXT:| |       |-MemberExpr {{.*}} <col:9> 'T__single':'T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| |       | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<T> *' implicit this
// CHECK-NEXT:| |       `-DeclRefExpr {{.*}} <col:16> 'T__unsafe_indexable':'T' lvalue ParmVar {{.*}} 'v' 'T__unsafe_indexable':'T'
// CHECK-NEXT:| |-ClassTemplateSpecialization {{.*}} 'RefParamMustBePtrGood'
// CHECK-NEXT:| |-ClassTemplateSpecializationDecl {{.*}} <line:4:1, line:37:1> line:5:7 class RefParamMustBePtrGood definition implicit_instantiation
// CHECK-NEXT:| | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-TemplateArgument type 'int *'
// CHECK-NEXT:| | | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:| | |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGood
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:6:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:7:5, col:16> col:16 ptr0 'int *__single':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:8:5, col:26> col:26 ptr1 'int *__unsafe_indexable':'int *'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:10:5, line:13:5> line:10:16 get_single 'int *() const__single' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:15:5, line:18:5> line:15:26 get_unsafe_indexable 'int *() const__unsafe_indexable' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:20:5, line:22:5> line:20:10 set_single 'void (int *__single)' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | | `-ParmVarDecl {{.*}} <col:21, col:32> col:32 v 'int *__single':'int *'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:24:5, line:27:5> line:24:10 set_single 'void ()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:29:5, line:32:5> line:29:10 set_unsafe_indexable 'void ()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:34:5, line:36:5> line:34:10 set_unsafe_indexable 'void (int *__unsafe_indexable)' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | | `-ParmVarDecl {{.*}} <col:31, col:52> col:52 v 'int *__unsafe_indexable':'int *'
// CHECK-NEXT:| | |-CXXConstructorDecl {{.*}} <line:5:7> col:7 implicit used RefParamMustBePtrGood 'void () noexcept' inline default trivial
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:| | |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGood 'void (const RefParamMustBePtrGood<int *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:| | | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamMustBePtrGood<int *> &'
// CHECK-NEXT:| | `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGood 'void (RefParamMustBePtrGood<int *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:| |   `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamMustBePtrGood<int *> &&'
// CHECK-NEXT:| `-ClassTemplateSpecializationDecl {{.*}} <line:4:1, line:37:1> line:5:7 class RefParamMustBePtrGood definition implicit_instantiation
// CHECK-NEXT:|   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-TemplateArgument type 'char *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'char *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'char'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGood
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:6:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:7:5, col:16> col:16 ptr0 'char *__single':'char *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:8:5, col:26> col:26 ptr1 'char *__unsafe_indexable':'char *'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:10:5, line:13:5> line:10:16 get_single 'char *() const__single' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:15:5, line:18:5> line:15:26 get_unsafe_indexable 'char *() const__unsafe_indexable' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:20:5, line:22:5> line:20:10 set_single 'void (char *__single)' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:21, col:32> col:32 v 'char *__single':'char *'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:24:5, line:27:5> line:24:10 set_single 'void ()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:29:5, line:32:5> line:29:10 set_unsafe_indexable 'void ()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:34:5, line:36:5> line:34:10 set_unsafe_indexable 'void (char *__unsafe_indexable)' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:31, col:52> col:52 v 'char *__unsafe_indexable':'char *'
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <line:5:7> col:7 implicit used RefParamMustBePtrGood 'void () noexcept' inline default trivial
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGood 'void (const RefParamMustBePtrGood<char *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamMustBePtrGood<char *> &'
// CHECK-NEXT:|   `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGood 'void (RefParamMustBePtrGood<char *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|     `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamMustBePtrGood<char *> &&'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:40:1, line:73:1> line:41:7 RefParamMustBePtrNeverInstantiated
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:40:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK-NEXT:| `-CXXRecordDecl {{.*}} <line:41:1, line:73:1> line:41:7 class RefParamMustBePtrNeverInstantiated definition
// CHECK-NEXT:|   |-DefinitionData aggregate standard_layout trivially_copyable trivial
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrNeverInstantiated
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:42:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:43:5, col:16> col:16 referenced ptr0 'T__single':'T'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:44:5, col:26> col:26 referenced ptr1 'T__unsafe_indexable':'T'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:46:5, line:49:5> line:46:16 get_single 'T () const__single' implicit-inline
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:35, line:49:5>
// CHECK-NEXT:|   |   |-DeclStmt {{.*}} <line:47:9, col:21>
// CHECK-NEXT:|   |   | `-VarDecl {{.*}} <col:9, col:17> col:11 referenced tmp 'T' nrvo cinit
// CHECK-NEXT:|   |   |   `-MemberExpr {{.*}} <col:17> 'T const__single':'const T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:|   |   |     `-CXXThisExpr {{.*}} <col:17> 'const RefParamMustBePtrNeverInstantiated<T> *' implicit this
// CHECK-NEXT:|   |   `-ReturnStmt {{.*}} <line:48:9, col:16> nrvo_candidate(Var {{.*}} 'tmp' 'T')
// CHECK-NEXT:|   |     `-DeclRefExpr {{.*}} <col:16> 'T' lvalue Var {{.*}} 'tmp' 'T'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:51:5, line:54:5> line:51:26 get_unsafe_indexable 'T () const__unsafe_indexable' implicit-inline
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:55, line:54:5>
// CHECK-NEXT:|   |   |-DeclStmt {{.*}} <line:52:9, col:21>
// CHECK-NEXT:|   |   | `-VarDecl {{.*}} <col:9, col:17> col:11 referenced tmp 'T' nrvo cinit
// CHECK-NEXT:|   |   |   `-MemberExpr {{.*}} <col:17> 'T const__unsafe_indexable':'const T' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:|   |   |     `-CXXThisExpr {{.*}} <col:17> 'const RefParamMustBePtrNeverInstantiated<T> *' implicit this
// CHECK-NEXT:|   |   `-ReturnStmt {{.*}} <line:53:9, col:16> nrvo_candidate(Var {{.*}} 'tmp' 'T')
// CHECK-NEXT:|   |     `-DeclRefExpr {{.*}} <col:16> 'T' lvalue Var {{.*}} 'tmp' 'T'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:56:5, line:58:5> line:56:10 set_single 'void (T__single)' implicit-inline
// CHECK-NEXT:|   | |-ParmVarDecl {{.*}} <col:21, col:32> col:32 referenced v 'T__single':'T'
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:35, line:58:5>
// CHECK-NEXT:|   |   `-BinaryOperator {{.*}} <line:57:9, col:16> '<dependent type>' '='
// CHECK-NEXT:|   |     |-MemberExpr {{.*}} <col:9> 'T__single':'T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:|   |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrNeverInstantiated<T> *' implicit this
// CHECK-NEXT:|   |     `-DeclRefExpr {{.*}} <col:16> 'T__single':'T' lvalue ParmVar {{.*}} 'v' 'T__single':'T'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:60:5, line:63:5> line:60:10 set_single 'void ()' implicit-inline
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:23, line:63:5>
// CHECK-NEXT:|   |   |-DeclStmt {{.*}} <line:61:9, col:33>
// CHECK-NEXT:|   |   | `-VarDecl {{.*}} <col:9, col:26> col:20 referenced tmp 'T__single':'T' cinit
// CHECK-NEXT:|   |   |   `-CXXNullPtrLiteralExpr {{.*}} <col:26> 'std::nullptr_t'
// CHECK-NEXT:|   |   `-BinaryOperator {{.*}} <line:62:9, col:16> '<dependent type>' '='
// CHECK-NEXT:|   |     |-MemberExpr {{.*}} <col:9> 'T__single':'T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:|   |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrNeverInstantiated<T> *' implicit this
// CHECK-NEXT:|   |     `-DeclRefExpr {{.*}} <col:16> 'T__single':'T' lvalue Var {{.*}} 'tmp' 'T__single':'T'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:65:5, line:68:5> line:65:10 set_unsafe_indexable 'void ()' implicit-inline
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:33, line:68:5>
// CHECK-NEXT:|   |   |-DeclStmt {{.*}} <line:66:9, col:43>
// CHECK-NEXT:|   |   | `-VarDecl {{.*}} <col:9, col:36> col:30 referenced tmp 'T__unsafe_indexable':'T' cinit
// CHECK-NEXT:|   |   |   `-CXXNullPtrLiteralExpr {{.*}} <col:36> 'std::nullptr_t'
// CHECK-NEXT:|   |   `-BinaryOperator {{.*}} <line:67:9, col:16> '<dependent type>' '='
// CHECK-NEXT:|   |     |-MemberExpr {{.*}} <col:9> 'T__unsafe_indexable':'T' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:|   |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrNeverInstantiated<T> *' implicit this
// CHECK-NEXT:|   |     `-DeclRefExpr {{.*}} <col:16> 'T__unsafe_indexable':'T' lvalue Var {{.*}} 'tmp' 'T__unsafe_indexable':'T'
// CHECK-NEXT:|   `-CXXMethodDecl {{.*}} <line:70:5, line:72:5> line:70:10 set_unsafe_indexable 'void (T__unsafe_indexable)' implicit-inline
// CHECK-NEXT:|     |-ParmVarDecl {{.*}} <col:31, col:52> col:52 referenced v 'T__unsafe_indexable':'T'
// CHECK-NEXT:|     `-CompoundStmt {{.*}} <col:55, line:72:5>
// CHECK-NEXT:|       `-BinaryOperator {{.*}} <line:71:9, col:16> '<dependent type>' '='
// CHECK-NEXT:|         |-MemberExpr {{.*}} <col:9> 'T__single':'T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:|         | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrNeverInstantiated<T> *' implicit this
// CHECK-NEXT:|         `-DeclRefExpr {{.*}} <col:16> 'T__unsafe_indexable':'T' lvalue ParmVar {{.*}} 'v' 'T__unsafe_indexable':'T'
// CHECK-NEXT:|-TypeAliasDecl {{.*}} <line:75:1, col:24> col:7 referenced PtrTypedef 'char *'
// CHECK-NEXT:| `-PointerType {{.*}} 'char *'
// CHECK-NEXT:|   `-BuiltinType {{.*}} 'char'
// CHECK-NEXT:|-ClassTemplateSpecializationDecl {{.*}} <line:78:1, line:79:35> col:7 class RefParamMustBePtrGood definition explicit_instantiation_definition
// CHECK-NEXT:| |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| |-TemplateArgument type 'float *'
// CHECK-NEXT:| | `-PointerType {{.*}} 'float *'
// CHECK-NEXT:| |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:5:1, col:7> col:7 implicit class RefParamMustBePtrGood
// CHECK-NEXT:| |-AccessSpecDecl {{.*}} <line:6:5, col:11> col:5 public
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:7:5, col:16> col:16 referenced ptr0 'float *__single':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:8:5, col:26> col:26 referenced ptr1 'float *__unsafe_indexable':'float *'
// CHECK-NEXT:| |-CXXMethodDecl {{.*}} <line:10:5, line:13:5> line:10:16 get_single 'float *() const__single' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | `-CompoundStmt {{.*}} <col:35, line:13:5>
// CHECK-NEXT:| |   |-DeclStmt {{.*}} <line:11:9, col:21>
// CHECK-NEXT:| |   | `-VarDecl {{.*}} <col:9, col:17> col:11 used tmp 'float *' cinit
// CHECK-NEXT:| |   |   `-ImplicitCastExpr {{.*}} <col:17> 'float *__single':'float *' <LValueToRValue>
// CHECK-NEXT:| |   |     `-MemberExpr {{.*}} <col:17> 'float *const__single':'float *const' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| |   |       `-CXXThisExpr {{.*}} <col:17> 'const RefParamMustBePtrGood<float *> *' implicit this
// CHECK-NEXT:| |   `-ReturnStmt {{.*}} <line:12:9, col:16>
// CHECK-NEXT:| |     `-ImplicitCastExpr {{.*}} <col:16> 'float *' <LValueToRValue>
// CHECK-NEXT:| |       `-DeclRefExpr {{.*}} <col:16> 'float *' lvalue Var {{.*}} 'tmp' 'float *'
// CHECK-NEXT:| |-CXXMethodDecl {{.*}} <line:15:5, line:18:5> line:15:26 get_unsafe_indexable 'float *() const__unsafe_indexable' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | `-CompoundStmt {{.*}} <col:55, line:18:5>
// CHECK-NEXT:| |   |-DeclStmt {{.*}} <line:16:9, col:21>
// CHECK-NEXT:| |   | `-VarDecl {{.*}} <col:9, col:17> col:11 used tmp 'float *' cinit
// CHECK-NEXT:| |   |   `-ImplicitCastExpr {{.*}} <col:17> 'float *__unsafe_indexable':'float *' <LValueToRValue>
// CHECK-NEXT:| |   |     `-MemberExpr {{.*}} <col:17> 'float *const__unsafe_indexable':'float *const' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:| |   |       `-CXXThisExpr {{.*}} <col:17> 'const RefParamMustBePtrGood<float *> *' implicit this
// CHECK-NEXT:| |   `-ReturnStmt {{.*}} <line:17:9, col:16>
// CHECK-NEXT:| |     `-ImplicitCastExpr {{.*}} <col:16> 'float *' <LValueToRValue>
// CHECK-NEXT:| |       `-DeclRefExpr {{.*}} <col:16> 'float *' lvalue Var {{.*}} 'tmp' 'float *'
// CHECK-NEXT:| |-CXXMethodDecl {{.*}} <line:20:5, line:22:5> line:20:10 set_single 'void (float *__single)' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-ParmVarDecl {{.*}} <col:21, col:32> col:32 used v 'float *__single':'float *'
// CHECK-NEXT:| | `-CompoundStmt {{.*}} <col:35, line:22:5>
// CHECK-NEXT:| |   `-BinaryOperator {{.*}} <line:21:9, col:16> 'float *__single':'float *' lvalue '='
// CHECK-NEXT:| |     |-MemberExpr {{.*}} <col:9> 'float *__single':'float *' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<float *> *' implicit this
// CHECK-NEXT:| |     `-ImplicitCastExpr {{.*}} <col:16> 'float *__single':'float *' <LValueToRValue>
// CHECK-NEXT:| |       `-DeclRefExpr {{.*}} <col:16> 'float *__single':'float *' lvalue ParmVar {{.*}} 'v' 'float *__single':'float *'
// CHECK-NEXT:| |-CXXMethodDecl {{.*}} <line:24:5, line:27:5> line:24:10 set_single 'void ()' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | `-CompoundStmt {{.*}} <col:23, line:27:5>
// CHECK-NEXT:| |   |-DeclStmt {{.*}} <line:25:9, col:33>
// CHECK-NEXT:| |   | `-VarDecl {{.*}} <col:9, col:26> col:20 used tmp 'float *__single':'float *' cinit
// CHECK-NEXT:| |   |   `-ImplicitCastExpr {{.*}} <col:26> 'float *__single':'float *' <NullToPointer>
// CHECK-NEXT:| |   |     `-CXXNullPtrLiteralExpr {{.*}} <col:26> 'std::nullptr_t'
// CHECK-NEXT:| |   `-BinaryOperator {{.*}} <line:26:9, col:16> 'float *__single':'float *' lvalue '='
// CHECK-NEXT:| |     |-MemberExpr {{.*}} <col:9> 'float *__single':'float *' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<float *> *' implicit this
// CHECK-NEXT:| |     `-ImplicitCastExpr {{.*}} <col:16> 'float *__single':'float *' <LValueToRValue>
// CHECK-NEXT:| |       `-DeclRefExpr {{.*}} <col:16> 'float *__single':'float *' lvalue Var {{.*}} 'tmp' 'float *__single':'float *'
// CHECK-NEXT:| |-CXXMethodDecl {{.*}} <line:29:5, line:32:5> line:29:10 set_unsafe_indexable 'void ()' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | `-CompoundStmt {{.*}} <col:33, line:32:5>
// CHECK-NEXT:| |   |-DeclStmt {{.*}} <line:30:9, col:43>
// CHECK-NEXT:| |   | `-VarDecl {{.*}} <col:9, col:36> col:30 used tmp 'float *__unsafe_indexable':'float *' cinit
// CHECK-NEXT:| |   |   `-ImplicitCastExpr {{.*}} <col:36> 'float *__unsafe_indexable':'float *' <NullToPointer>
// CHECK-NEXT:| |   |     `-CXXNullPtrLiteralExpr {{.*}} <col:36> 'std::nullptr_t'
// CHECK-NEXT:| |   `-BinaryOperator {{.*}} <line:31:9, col:16> 'float *__unsafe_indexable':'float *' lvalue '='
// CHECK-NEXT:| |     |-MemberExpr {{.*}} <col:9> 'float *__unsafe_indexable':'float *' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:| |     | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<float *> *' implicit this
// CHECK-NEXT:| |     `-ImplicitCastExpr {{.*}} <col:16> 'float *__unsafe_indexable':'float *' <LValueToRValue>
// CHECK-NEXT:| |       `-DeclRefExpr {{.*}} <col:16> 'float *__unsafe_indexable':'float *' lvalue Var {{.*}} 'tmp' 'float *__unsafe_indexable':'float *'
// CHECK-NEXT:| `-CXXMethodDecl {{.*}} <line:34:5, line:36:5> line:34:10 set_unsafe_indexable 'void (float *__unsafe_indexable)' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-ParmVarDecl {{.*}} <col:31, col:52> col:52 used v 'float *__unsafe_indexable':'float *'
// CHECK-NEXT:|   `-CompoundStmt {{.*}} <col:55, line:36:5>
// CHECK-NEXT:|     `-BinaryOperator {{.*}} <line:35:9, col:16> 'float *__single':'float *' lvalue '='
// CHECK-NEXT:|       |-MemberExpr {{.*}} <col:9> 'float *__single':'float *' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:|       | `-CXXThisExpr {{.*}} <col:9> 'RefParamMustBePtrGood<float *> *' implicit this
// CHECK-NEXT:|       `-ImplicitCastExpr {{.*}} <col:16> 'float *__unsafe_indexable':'float *' <LValueToRValue>
// CHECK-NEXT:|         `-DeclRefExpr {{.*}} <col:16> 'float *__unsafe_indexable':'float *' lvalue ParmVar {{.*}} 'v' 'float *__unsafe_indexable':'float *'
// CHECK-NEXT:|-FunctionDecl {{.*}} <line:81:1, line:85:1> line:81:6 Instantiate_RefParamMustBePtrGood 'void ()'
// CHECK-NEXT:| `-CompoundStmt {{.*}} <col:42, line:85:1>
// CHECK-NEXT:|   |-DeclStmt {{.*}} <line:83:5, col:38>
// CHECK-NEXT:|   | `-VarDecl {{.*}} <col:5, col:33> col:33 good0 'RefParamMustBePtrGood<int *>' callinit
// CHECK-NEXT:|   |   `-CXXConstructExpr {{.*}} <col:33> 'RefParamMustBePtrGood<int *>' 'void () noexcept'
// CHECK-NEXT:|   `-DeclStmt {{.*}} <line:84:5, col:44>
// CHECK-NEXT:|     `-VarDecl {{.*}} <col:5, col:39> col:39 good1 'RefParamMustBePtrGood<PtrTypedef>':'RefParamMustBePtrGood<char *>' callinit
// CHECK-NEXT:|       `-CXXConstructExpr {{.*}} <col:39> 'RefParamMustBePtrGood<PtrTypedef>':'RefParamMustBePtrGood<char *>' 'void () noexcept'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:88:1, line:93:1> line:89:7 RefParamIsPointee
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:88:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:89:1, line:93:1> line:89:7 class RefParamIsPointee definition
// CHECK-NEXT:| | |-DefinitionData aggregate standard_layout trivially_copyable pod trivial
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamIsPointee
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:90:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:91:5, col:17> col:17 ptr0 'T *__single':'T *'
// CHECK-NEXT:| | `-FieldDecl {{.*}} <line:92:5, col:27> col:27 pt1 'T *__unsafe_indexable':'T *'
// CHECK-NEXT:| |-ClassTemplateSpecialization {{.*}} 'RefParamIsPointee'
// CHECK-NEXT:| |-ClassTemplateSpecializationDecl {{.*}} <line:88:1, line:93:1> line:89:7 class RefParamIsPointee definition implicit_instantiation
// CHECK-NEXT:| | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-TemplateArgument type 'int'
// CHECK-NEXT:| | | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamIsPointee
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:90:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:91:5, col:17> col:17 ptr0 'int *__single':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:92:5, col:27> col:27 pt1 'int *__unsafe_indexable':'int *'
// CHECK-NEXT:| | |-CXXConstructorDecl {{.*}} <line:89:7> col:7 implicit used RefParamIsPointee 'void () noexcept' inline default trivial
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:| | |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamIsPointee 'void (const RefParamIsPointee<int> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:| | | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamIsPointee<int> &'
// CHECK-NEXT:| | `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamIsPointee 'void (RefParamIsPointee<int> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:| |   `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamIsPointee<int> &&'
// CHECK-NEXT:| `-ClassTemplateSpecializationDecl {{.*}} <line:88:1, line:93:1> line:89:7 class RefParamIsPointee definition implicit_instantiation
// CHECK-NEXT:|   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-TemplateArgument type 'char *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'char *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'char'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamIsPointee
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:90:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:91:5, col:17> col:17 ptr0 'char **__single':'char **'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:92:5, col:27> col:27 pt1 'char **__unsafe_indexable':'char **'
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <line:89:7> col:7 implicit used RefParamIsPointee 'void () noexcept' inline default trivial
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamIsPointee 'void (const RefParamIsPointee<char *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamIsPointee<char *> &'
// CHECK-NEXT:|   `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamIsPointee 'void (RefParamIsPointee<char *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|     `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamIsPointee<char *> &&'
// CHECK-NEXT:|-ClassTemplateSpecializationDecl {{.*}} <line:96:1, col:39> col:16 class RefParamIsPointee definition explicit_instantiation_definition
// CHECK-NEXT:| |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| |-TemplateArgument type 'float'
// CHECK-NEXT:| | `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:89:1, col:7> col:7 implicit class RefParamIsPointee
// CHECK-NEXT:| |-AccessSpecDecl {{.*}} <line:90:5, col:11> col:5 public
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:91:5, col:17> col:17 ptr0 'float *__single':'float *'
// CHECK-NEXT:| `-FieldDecl {{.*}} <line:92:5, col:27> col:27 pt1 'float *__unsafe_indexable':'float *'
// CHECK-NEXT:|-FunctionDecl {{.*}} <line:98:1, line:102:1> line:98:6 Instantiate_RefParamIsPointee 'void ()'
// CHECK-NEXT:| `-CompoundStmt {{.*}} <col:38, line:102:1>
// CHECK-NEXT:|   |-DeclStmt {{.*}} <line:100:5, col:33>
// CHECK-NEXT:|   | `-VarDecl {{.*}} <col:5, col:28> col:28 good0 'RefParamIsPointee<int>' callinit
// CHECK-NEXT:|   |   `-CXXConstructExpr {{.*}} <col:28> 'RefParamIsPointee<int>' 'void () noexcept'
// CHECK-NEXT:|   `-DeclStmt {{.*}} <line:101:5, col:40>
// CHECK-NEXT:|     `-VarDecl {{.*}} <col:5, col:35> col:35 good1 'RefParamIsPointee<PtrTypedef>':'RefParamIsPointee<char *>' callinit
// CHECK-NEXT:|       `-CXXConstructExpr {{.*}} <col:35> 'RefParamIsPointee<PtrTypedef>':'RefParamIsPointee<char *>' 'void () noexcept'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:107:1, line:114:1> line:108:7 TInMethodBodyGood
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:107:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:108:1, line:114:1> line:108:7 class TInMethodBodyGood definition
// CHECK-NEXT:| | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class TInMethodBodyGood
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:109:5, col:11> col:5 public
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:110:5, line:113:5> line:110:10 test 'void ()' implicit-inline
// CHECK-NEXT:| |   `-CompoundStmt {{.*}} <col:17, line:113:5>
// CHECK-NEXT:| |     |-DeclStmt {{.*}} <line:111:9, col:23>
// CHECK-NEXT:| |     | `-VarDecl {{.*}} <col:9, col:20> col:20 referenced tmp 'T__single':'T'
// CHECK-NEXT:| |     `-BinaryOperator {{.*}} <line:112:9, col:15> '<dependent type>' '='
// CHECK-NEXT:| |       |-DeclRefExpr {{.*}} <col:9> 'T__single':'T' lvalue Var {{.*}} 'tmp' 'T__single':'T'
// CHECK-NEXT:| |       `-CXXNullPtrLiteralExpr {{.*}} <col:15> 'std::nullptr_t'
// CHECK-NEXT:| |-ClassTemplateSpecialization {{.*}} 'TInMethodBodyGood'
// CHECK-NEXT:| `-ClassTemplateSpecializationDecl {{.*}} <line:107:1, line:114:1> line:108:7 class TInMethodBodyGood definition implicit_instantiation
// CHECK-NEXT:|   |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial constexpr defaulted_is_constexpr
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-TemplateArgument type 'int *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class TInMethodBodyGood
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:109:5, col:11> col:5 public
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:110:5, line:113:5> line:110:10 used test 'void ()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:17, line:113:5>
// CHECK-NEXT:|   |   |-DeclStmt {{.*}} <line:111:9, col:23>
// CHECK-NEXT:|   |   | `-VarDecl {{.*}} <col:9, col:20> col:20 used tmp 'int *__single':'int *'
// CHECK-NEXT:|   |   `-BinaryOperator {{.*}} <line:112:9, col:15> 'int *__single':'int *' lvalue '='
// CHECK-NEXT:|   |     |-DeclRefExpr {{.*}} <col:9> 'int *__single':'int *' lvalue Var {{.*}} 'tmp' 'int *__single':'int *'
// CHECK-NEXT:|   |     `-ImplicitCastExpr {{.*}} <col:15> 'int *__single':'int *' <NullToPointer>
// CHECK-NEXT:|   |       `-CXXNullPtrLiteralExpr {{.*}} <col:15> 'std::nullptr_t'
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <line:108:7> col:7 implicit used constexpr TInMethodBodyGood 'void () noexcept' inline default trivial
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr TInMethodBodyGood 'void (const TInMethodBodyGood<int *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:7> col:7 'const TInMethodBodyGood<int *> &'
// CHECK-NEXT:|   `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr TInMethodBodyGood 'void (TInMethodBodyGood<int *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|     `-ParmVarDecl {{.*}} <col:7> col:7 'TInMethodBodyGood<int *> &&'
// CHECK-NEXT:|-ClassTemplateSpecializationDecl {{.*}} <line:117:1, col:40> col:16 class TInMethodBodyGood definition explicit_instantiation_definition
// CHECK-NEXT:| |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT:| | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT:| | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| |-TemplateArgument type 'float *'
// CHECK-NEXT:| | `-PointerType {{.*}} 'float *'
// CHECK-NEXT:| |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:108:1, col:7> col:7 implicit class TInMethodBodyGood
// CHECK-NEXT:| |-AccessSpecDecl {{.*}} <line:109:5, col:11> col:5 public
// CHECK-NEXT:| `-CXXMethodDecl {{.*}} <line:110:5, line:113:5> line:110:10 test 'void ()' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   `-CompoundStmt {{.*}} <col:17, line:113:5>
// CHECK-NEXT:|     |-DeclStmt {{.*}} <line:111:9, col:23>
// CHECK-NEXT:|     | `-VarDecl {{.*}} <col:9, col:20> col:20 used tmp 'float *__single':'float *'
// CHECK-NEXT:|     `-BinaryOperator {{.*}} <line:112:9, col:15> 'float *__single':'float *' lvalue '='
// CHECK-NEXT:|       |-DeclRefExpr {{.*}} <col:9> 'float *__single':'float *' lvalue Var {{.*}} 'tmp' 'float *__single':'float *'
// CHECK-NEXT:|       `-ImplicitCastExpr {{.*}} <col:15> 'float *__single':'float *' <NullToPointer>
// CHECK-NEXT:|         `-CXXNullPtrLiteralExpr {{.*}} <col:15> 'std::nullptr_t'
// CHECK-NEXT:|-FunctionDecl {{.*}} <line:119:1, line:123:1> line:119:6 Instantiate_TInMethodBodyGood 'void ()'
// CHECK-NEXT:| `-CompoundStmt {{.*}} <col:38, line:123:1>
// CHECK-NEXT:|   |-DeclStmt {{.*}} <line:121:5, col:34>
// CHECK-NEXT:|   | `-VarDecl {{.*}} <col:5, col:29> col:29 used good0 'TInMethodBodyGood<int *>' callinit
// CHECK-NEXT:|   |   `-CXXConstructExpr {{.*}} <col:29> 'TInMethodBodyGood<int *>' 'void () noexcept'
// CHECK-NEXT:|   `-CXXMemberCallExpr {{.*}} <line:122:5, col:16> 'void'
// CHECK-NEXT:|     `-MemberExpr {{.*}} <col:5, col:11> '<bound member function type>' .test {{.*}}
// CHECK-NEXT:|       `-DeclRefExpr {{.*}} <col:5> 'TInMethodBodyGood<int *>' lvalue Var {{.*}} 'good0' 'TInMethodBodyGood<int *>'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:131:1, line:149:1> line:132:7 RefParamMustBePtrGoodPartialBase
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:131:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <col:20, col:26> col:26 referenced class depth 0 index 1 U
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <col:29, col:35> col:35 referenced class depth 0 index 2 V
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:132:1, line:149:1> line:132:7 class RefParamMustBePtrGoodPartialBase definition
// CHECK-NEXT:| | |-DefinitionData aggregate standard_layout trivially_copyable trivial
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialBase
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:133:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:134:5, col:16> col:16 referenced ptr0 'T__single':'T'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:135:5, col:26> col:26 referenced ptr1 'U__unsafe_indexable':'U'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:136:5, col:7> col:7 referenced counter 'V'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:138:5, col:48> col:16 get_ptr0 'T () const__single' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:33, col:48>
// CHECK-NEXT:| | |   `-ReturnStmt {{.*}} <col:35, col:42>
// CHECK-NEXT:| | |     `-MemberExpr {{.*}} <col:42> 'T const__single':'const T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| | |       `-CXXThisExpr {{.*}} <col:42> 'const RefParamMustBePtrGoodPartialBase<T, U, V> *' implicit this
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:139:5, col:58> col:26 get_ptr1 'U () const__unsafe_indexable' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:43, col:58>
// CHECK-NEXT:| | |   `-ReturnStmt {{.*}} <col:45, col:52>
// CHECK-NEXT:| | |     `-MemberExpr {{.*}} <col:52> 'U const__unsafe_indexable':'const U' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:| | |       `-CXXThisExpr {{.*}} <col:52> 'const RefParamMustBePtrGoodPartialBase<T, U, V> *' implicit this
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:140:5, col:45> col:7 get_counter 'V () const' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:27, col:45>
// CHECK-NEXT:| | |   `-ReturnStmt {{.*}} <col:29, col:36>
// CHECK-NEXT:| | |     `-MemberExpr {{.*}} <col:36> 'const V' lvalue ->counter {{.*}}
// CHECK-NEXT:| | |       `-CXXThisExpr {{.*}} <col:36> 'const RefParamMustBePtrGoodPartialBase<T, U, V> *' implicit this
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:142:5, line:144:5> line:142:10 useT 'void () const' implicit-inline
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:23, line:144:5>
// CHECK-NEXT:| | |   `-DeclStmt {{.*}} <line:143:9, col:30>
// CHECK-NEXT:| | |     `-VarDecl {{.*}} <col:9, col:26> col:20 tmp 'T__single':'T' cinit
// CHECK-NEXT:| | |       `-MemberExpr {{.*}} <col:26> 'T const__single':'const T' lvalue ->ptr0 {{.*}}
// CHECK-NEXT:| | |         `-CXXThisExpr {{.*}} <col:26> 'const RefParamMustBePtrGoodPartialBase<T, U, V> *' implicit this
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:146:5, line:148:5> line:146:10 useU 'void () const' implicit-inline
// CHECK-NEXT:| |   `-CompoundStmt {{.*}} <col:23, line:148:5>
// CHECK-NEXT:| |     `-DeclStmt {{.*}} <line:147:9, col:40>
// CHECK-NEXT:| |       `-VarDecl {{.*}} <col:9, col:36> col:30 tmp 'T__unsafe_indexable':'T' cinit
// CHECK-NEXT:| |         `-MemberExpr {{.*}} <col:36> 'U const__unsafe_indexable':'const U' lvalue ->ptr1 {{.*}}
// CHECK-NEXT:| |           `-CXXThisExpr {{.*}} <col:36> 'const RefParamMustBePtrGoodPartialBase<T, U, V> *' implicit this
// CHECK-NEXT:| |-ClassTemplateSpecializationDecl {{.*}} <line:131:1, line:149:1> line:132:7 class RefParamMustBePtrGoodPartialBase definition implicit_instantiation
// CHECK-NEXT:| | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-TemplateArgument type 'float *'
// CHECK-NEXT:| | | `-PointerType {{.*}} 'float *'
// CHECK-NEXT:| | |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| | |-TemplateArgument type 'int *'
// CHECK-NEXT:| | | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:| | |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:| | |-TemplateArgument type 'int'
// CHECK-NEXT:| | | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialBase
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:133:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:134:5, col:16> col:16 referenced ptr0 'float *__single':'float *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:135:5, col:26> col:26 ptr1 'int *__unsafe_indexable':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:136:5, col:7> col:7 counter 'int'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:138:5, col:48> col:16 get_ptr0 'float *() const__single' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:139:5, col:58> col:26 get_ptr1 'int *() const__unsafe_indexable' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:140:5, col:45> col:7 get_counter 'int () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:142:5, line:144:5> line:142:10 useT 'void () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:146:5, line:148:5> line:146:10 useU 'void () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| |-ClassTemplateSpecializationDecl {{.*}} <line:131:1, line:149:1> line:132:7 class RefParamMustBePtrGoodPartialBase definition implicit_instantiation
// CHECK-NEXT:| | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-TemplateArgument type 'int *'
// CHECK-NEXT:| | | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:| | |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:| | |-TemplateArgument type 'float *'
// CHECK-NEXT:| | | `-PointerType {{.*}} 'float *'
// CHECK-NEXT:| | |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| | |-TemplateArgument type 'int'
// CHECK-NEXT:| | | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialBase
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:133:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:134:5, col:16> col:16 ptr0 'int *__single':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:135:5, col:26> col:26 referenced ptr1 'float *__unsafe_indexable':'float *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:136:5, col:7> col:7 counter 'int'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:138:5, col:48> col:16 get_ptr0 'int *() const__single' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:139:5, col:58> col:26 get_ptr1 'float *() const__unsafe_indexable' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:140:5, col:45> col:7 get_counter 'int () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:142:5, line:144:5> line:142:10 useT 'void () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:146:5, line:148:5> line:146:10 useU 'void () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| `-ClassTemplateSpecializationDecl {{.*}} <line:131:1, line:149:1> line:132:7 class RefParamMustBePtrGoodPartialBase definition implicit_instantiation
// CHECK-NEXT:|   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial
// CHECK-NEXT:|   |-TemplateArgument type 'int *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:|   |-TemplateArgument type 'int *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:|   |-TemplateArgument type 'int'
// CHECK-NEXT:|   | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialBase
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:133:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:134:5, col:16> col:16 referenced ptr0 'int *__single':'int *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:135:5, col:26> col:26 referenced ptr1 'int *__unsafe_indexable':'int *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:136:5, col:7> col:7 counter 'int'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:138:5, col:48> col:16 get_ptr0 'int *() const__single' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:139:5, col:58> col:26 get_ptr1 'int *() const__unsafe_indexable' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:140:5, col:45> col:7 get_counter 'int () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:142:5, line:144:5> line:142:10 useT 'void () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:146:5, line:148:5> line:146:10 useU 'void () const' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <line:132:7> col:7 implicit used RefParamMustBePtrGoodPartialBase 'void () noexcept' inline default trivial
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:|   |-CXXDestructorDecl {{.*}} <col:7> col:7 implicit ~RefParamMustBePtrGoodPartialBase 'void ()' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGoodPartialBase 'void (const RefParamMustBePtrGoodPartialBase<int *, int *, int> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamMustBePtrGoodPartialBase<int *, int *, int> &'
// CHECK-NEXT:|   `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGoodPartialBase 'void (RefParamMustBePtrGoodPartialBase<int *, int *, int> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|     `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamMustBePtrGoodPartialBase<int *, int *, int> &&'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:152:1, line:158:1> line:153:7 RefParamMustBePtrGoodPartialT
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:152:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:153:1, line:158:1> line:153:7 class RefParamMustBePtrGoodPartialT definition
// CHECK-NEXT:| | |-DefinitionData aggregate standard_layout trivially_copyable trivial
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-public 'RefParamMustBePtrGoodPartialBase<T, int *, int>'
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialT
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:154:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:155:5, col:16> col:16 referenced ptr2 'T__single':'T'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:156:5, col:66> col:66 ptr3 'typeof (RefParamMustBePtrGoodPartialBase<T, int *, int>::ptr0)'
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:157:5, col:48> col:16 another_method 'T ()__single' implicit-inline
// CHECK-NEXT:| |   `-CompoundStmt {{.*}} <col:33, col:48>
// CHECK-NEXT:| |     `-ReturnStmt {{.*}} <col:35, col:42>
// CHECK-NEXT:| |       `-MemberExpr {{.*}} <col:42> 'T__single':'T' lvalue ->ptr2 {{.*}}
// CHECK-NEXT:| |         `-CXXThisExpr {{.*}} <col:42> 'RefParamMustBePtrGoodPartialT<T> *' implicit this
// CHECK-NEXT:| |-ClassTemplateSpecialization {{.*}} 'RefParamMustBePtrGoodPartialT'
// CHECK-NEXT:| `-ClassTemplateSpecializationDecl {{.*}} <line:152:1, line:158:1> line:153:7 class RefParamMustBePtrGoodPartialT definition implicit_instantiation
// CHECK-NEXT:|   |-DefinitionData pass_in_registers aggregate trivially_copyable trivial literal
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-public 'RefParamMustBePtrGoodPartialBase<int *, int *, int>'
// CHECK-NEXT:|   |-TemplateArgument type 'int *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialT
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:154:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:155:5, col:16> col:16 ptr2 'int *__single':'int *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:156:5, col:66> col:66 ptr3 'typeof (RefParamMustBePtrGoodPartialBase<int *, int *, int>::ptr0)':'int *'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:157:5, col:48> col:16 another_method 'int *()__single' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <line:153:7> col:7 implicit used RefParamMustBePtrGoodPartialT 'void () noexcept' inline default trivial
// CHECK-NEXT:|   | |-CXXCtorInitializer 'RefParamMustBePtrGoodPartialBase<int *, int *, int>'
// CHECK-NEXT:|   | | `-CXXConstructExpr {{.*}} <col:7> 'RefParamMustBePtrGoodPartialBase<int *, int *, int>' 'void () noexcept'
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGoodPartialT 'void (const RefParamMustBePtrGoodPartialT<int *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamMustBePtrGoodPartialT<int *> &'
// CHECK-NEXT:|   `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGoodPartialT 'void (RefParamMustBePtrGoodPartialT<int *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|     `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamMustBePtrGoodPartialT<int *> &&'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:160:1, line:166:1> line:161:7 RefParamMustBePtrGoodPartialU
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:160:11, col:17> col:17 referenced class depth 0 index 0 U
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:161:1, line:166:1> line:161:7 class RefParamMustBePtrGoodPartialU definition
// CHECK-NEXT:| | |-DefinitionData aggregate standard_layout trivially_copyable trivial
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-public 'RefParamMustBePtrGoodPartialBase<int *, U, int>'
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialU
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:162:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:163:5, col:26> col:26 referenced ptr2 'U__unsafe_indexable':'U'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:164:5, col:66> col:66 ptr3 'typeof (RefParamMustBePtrGoodPartialBase<int *, U, int>::ptr1)'
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:165:5, col:58> col:26 another_method 'U ()__unsafe_indexable' implicit-inline
// CHECK-NEXT:| |   `-CompoundStmt {{.*}} <col:43, col:58>
// CHECK-NEXT:| |     `-ReturnStmt {{.*}} <col:45, col:52>
// CHECK-NEXT:| |       `-MemberExpr {{.*}} <col:52> 'U__unsafe_indexable':'U' lvalue ->ptr2 {{.*}}
// CHECK-NEXT:| |         `-CXXThisExpr {{.*}} <col:52> 'RefParamMustBePtrGoodPartialU<U> *' implicit this
// CHECK-NEXT:| |-ClassTemplateSpecialization {{.*}} 'RefParamMustBePtrGoodPartialU'
// CHECK-NEXT:| `-ClassTemplateSpecializationDecl {{.*}} <line:160:1, line:166:1> line:161:7 class RefParamMustBePtrGoodPartialU definition implicit_instantiation
// CHECK-NEXT:|   |-DefinitionData pass_in_registers aggregate trivially_copyable trivial literal
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-public 'RefParamMustBePtrGoodPartialBase<int *, int *, int>'
// CHECK-NEXT:|   |-TemplateArgument type 'int *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialU
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:162:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:163:5, col:26> col:26 ptr2 'int *__unsafe_indexable':'int *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:164:5, col:66> col:66 ptr3 'typeof (RefParamMustBePtrGoodPartialBase<int *, int *, int>::ptr1)':'int *'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:165:5, col:58> col:26 another_method 'int *()__unsafe_indexable' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <line:161:7> col:7 implicit used RefParamMustBePtrGoodPartialU 'void () noexcept' inline default trivial
// CHECK-NEXT:|   | |-CXXCtorInitializer 'RefParamMustBePtrGoodPartialBase<int *, int *, int>'
// CHECK-NEXT:|   | | `-CXXConstructExpr {{.*}} <col:7> 'RefParamMustBePtrGoodPartialBase<int *, int *, int>' 'void () noexcept'
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGoodPartialU 'void (const RefParamMustBePtrGoodPartialU<int *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamMustBePtrGoodPartialU<int *> &'
// CHECK-NEXT:|   `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrGoodPartialU 'void (RefParamMustBePtrGoodPartialU<int *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|     `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamMustBePtrGoodPartialU<int *> &&'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:169:1, line:175:1> line:170:7 RefParamMustBePtrGoodPartialV
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:169:11, col:17> col:17 referenced class depth 0 index 0 V
// CHECK-NEXT:| `-CXXRecordDecl {{.*}} <line:170:1, line:175:1> line:170:7 class RefParamMustBePtrGoodPartialV definition
// CHECK-NEXT:|   |-DefinitionData aggregate standard_layout trivially_copyable trivial
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-public 'RefParamMustBePtrGoodPartialBase<int, int, V>'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialV
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:171:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:172:5, col:16> col:16 referenced ptr2 'V__single':'V'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:173:5, col:65> col:65 ptr3 'typeof (RefParamMustBePtrGoodPartialBase<int, int, V>::ptr0)'
// CHECK-NEXT:|   `-CXXMethodDecl {{.*}} <line:174:5, col:48> col:16 another_method 'V ()__single' implicit-inline
// CHECK-NEXT:|     `-CompoundStmt {{.*}} <col:33, col:48>
// CHECK-NEXT:|       `-ReturnStmt {{.*}} <col:35, col:42>
// CHECK-NEXT:|         `-MemberExpr {{.*}} <col:42> 'V__single':'V' lvalue ->ptr2 {{.*}}
// CHECK-NEXT:|           `-CXXThisExpr {{.*}} <col:42> 'RefParamMustBePtrGoodPartialV<V> *' implicit this
// CHECK-NEXT:|-ClassTemplateSpecializationDecl {{.*}} <line:178:1, line:179:43> col:7 class RefParamMustBePtrGoodPartialT definition explicit_instantiation_definition
// CHECK-NEXT:| |-DefinitionData pass_in_registers aggregate trivially_copyable trivial literal
// CHECK-NEXT:| | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| |-public 'RefParamMustBePtrGoodPartialBase<float *, int *, int>'
// CHECK-NEXT:| |-TemplateArgument type 'float *'
// CHECK-NEXT:| | `-PointerType {{.*}} 'float *'
// CHECK-NEXT:| |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:153:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialT
// CHECK-NEXT:| |-AccessSpecDecl {{.*}} <line:154:5, col:11> col:5 public
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:155:5, col:16> col:16 referenced ptr2 'float *__single':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:156:5, col:66> col:66 ptr3 'typeof (RefParamMustBePtrGoodPartialBase<float *, int *, int>::ptr0)':'float *'
// CHECK-NEXT:| `-CXXMethodDecl {{.*}} <line:157:5, col:48> col:16 another_method 'float *()__single' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   `-CompoundStmt {{.*}} <col:33, col:48>
// CHECK-NEXT:|     `-ReturnStmt {{.*}} <col:35, col:42>
// CHECK-NEXT:|       `-ImplicitCastExpr {{.*}} <col:42> 'float *__single':'float *' <LValueToRValue>
// CHECK-NEXT:|         `-MemberExpr {{.*}} <col:42> 'float *__single':'float *' lvalue ->ptr2 {{.*}}
// CHECK-NEXT:|           `-CXXThisExpr {{.*}} <col:42> 'RefParamMustBePtrGoodPartialT<float *> *' implicit this
// CHECK-NEXT:|-ClassTemplateSpecializationDecl {{.*}} <line:180:1, line:181:43> col:7 class RefParamMustBePtrGoodPartialU definition explicit_instantiation_definition
// CHECK-NEXT:| |-DefinitionData pass_in_registers aggregate trivially_copyable trivial literal
// CHECK-NEXT:| | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| |-public 'RefParamMustBePtrGoodPartialBase<int *, float *, int>'
// CHECK-NEXT:| |-TemplateArgument type 'float *'
// CHECK-NEXT:| | `-PointerType {{.*}} 'float *'
// CHECK-NEXT:| |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:161:1, col:7> col:7 implicit class RefParamMustBePtrGoodPartialU
// CHECK-NEXT:| |-AccessSpecDecl {{.*}} <line:162:5, col:11> col:5 public
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:163:5, col:26> col:26 referenced ptr2 'float *__unsafe_indexable':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:164:5, col:66> col:66 ptr3 'typeof (RefParamMustBePtrGoodPartialBase<int *, float *, int>::ptr1)':'float *'
// CHECK-NEXT:| `-CXXMethodDecl {{.*}} <line:165:5, col:58> col:26 another_method 'float *()__unsafe_indexable' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   `-CompoundStmt {{.*}} <col:43, col:58>
// CHECK-NEXT:|     `-ReturnStmt {{.*}} <col:45, col:52>
// CHECK-NEXT:|       `-ImplicitCastExpr {{.*}} <col:52> 'float *__unsafe_indexable':'float *' <LValueToRValue>
// CHECK-NEXT:|         `-MemberExpr {{.*}} <col:52> 'float *__unsafe_indexable':'float *' lvalue ->ptr2 {{.*}}
// CHECK-NEXT:|           `-CXXThisExpr {{.*}} <col:52> 'RefParamMustBePtrGoodPartialU<float *> *' implicit this
// CHECK-NEXT:|-FunctionDecl {{.*}} <line:183:1, line:187:1> line:183:6 Instantiate_RefParamMustBePtrGoodPartial 'void ()'
// CHECK-NEXT:| `-CompoundStmt {{.*}} <col:49, line:187:1>
// CHECK-NEXT:|   |-DeclStmt {{.*}} <line:185:5, col:46>
// CHECK-NEXT:|   | `-VarDecl {{.*}} <col:5, col:41> col:41 good0 'RefParamMustBePtrGoodPartialT<int *>' callinit
// CHECK-NEXT:|   |   `-CXXConstructExpr {{.*}} <col:41> 'RefParamMustBePtrGoodPartialT<int *>' 'void () noexcept'
// CHECK-NEXT:|   `-DeclStmt {{.*}} <line:186:5, col:46>
// CHECK-NEXT:|     `-VarDecl {{.*}} <col:5, col:41> col:41 good1 'RefParamMustBePtrGoodPartialU<int *>' callinit
// CHECK-NEXT:|       `-CXXConstructExpr {{.*}} <col:41> 'RefParamMustBePtrGoodPartialU<int *>' 'void () noexcept'
// CHECK-NEXT:|-ClassTemplateDecl {{.*}} <line:194:1, line:218:1> line:195:7 RefParamMustBePtrExternallyCountedGood
// CHECK-NEXT:| |-TemplateTypeParmDecl {{.*}} <line:194:11, col:17> col:17 referenced class depth 0 index 0 T
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:195:1, line:218:1> line:195:7 class RefParamMustBePtrExternallyCountedGood definition
// CHECK-NEXT:| | |-DefinitionData aggregate standard_layout trivially_copyable trivial
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrExternallyCountedGood
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:196:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:197:5, col:9> col:9 referenced size 'int'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:198:5, col:7> col:7 referenced end_ptr 'T'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:199:5, col:26> col:26 referenced cb 'T'
// CHECK-NEXT:| | | `-CountedByAttr {{.*}} <{{.+}}ptrcheck.h:56:40, col:56> 0
// CHECK-NEXT:| | |   `-MemberExpr {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:199:20> 'int' lvalue ->size {{.*}}
// CHECK-NEXT:| | |     `-CXXThisExpr {{.*}} <col:20> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:200:5, col:34> col:34 cbon 'T'
// CHECK-NEXT:| | | `-CountedByOrNullAttr {{.*}} <{{.+}}ptrcheck.h:60:48, col:72> 0
// CHECK-NEXT:| | |   `-MemberExpr {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:200:28> 'int' lvalue ->size {{.*}}
// CHECK-NEXT:| | |     `-CXXThisExpr {{.*}} <col:28> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:201:5, col:24> col:24 sb 'T'
// CHECK-NEXT:| | | `-SizedByAttr {{.*}} <{{.+}}ptrcheck.h:64:38, col:52> 0
// CHECK-NEXT:| | |   `-MemberExpr {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:201:18> 'int' lvalue ->size {{.*}}
// CHECK-NEXT:| | |     `-CXXThisExpr {{.*}} <col:18> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:202:5, col:32> col:32 sbon 'T'
// CHECK-NEXT:| | | `-SizedByOrNullAttr {{.*}} <{{.+}}ptrcheck.h:68:46, col:68> 0
// CHECK-NEXT:| | |   `-MemberExpr {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:202:26> 'int' lvalue ->size {{.*}}
// CHECK-NEXT:| | |     `-CXXThisExpr {{.*}} <col:26> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:203:5, col:27> col:27 eb 'T'
// CHECK-NEXT:| | | `-PtrEndedByAttr {{.*}} <{{.+}}ptrcheck.h:80:38, col:52> 0
// CHECK-NEXT:| | |   `-MemberExpr {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:203:18> 'T' lvalue ->end_ptr {{.*}}
// CHECK-NEXT:| | |     `-CXXThisExpr {{.*}} <col:18> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:205:5, line:207:5> line:205:26 ret_cb 'T ()' implicit-inline
// CHECK-NEXT:| | | |-CompoundStmt {{.*}} <col:35, line:207:5>
// CHECK-NEXT:| | | | `-ReturnStmt {{.*}} <line:206:9, col:16>
// CHECK-NEXT:| | | |   `-MemberExpr {{.*}} <col:16> 'T' lvalue ->cb {{.*}}
// CHECK-NEXT:| | | |     `-CXXThisExpr {{.*}} <col:16> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| | | `-CountedByAttr {{.*}} <{{.+}}ptrcheck.h:56:40, col:56> 0
// CHECK-NEXT:| | |   `-MemberExpr {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:205:20> 'int' lvalue ->size {{.*}}
// CHECK-NEXT:| | |     `-CXXThisExpr {{.*}} <col:20> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| | `-CXXMethodDecl {{.*}} <line:214:5, line:217:5> line:214:10 useT 'void ()' implicit-inline
// CHECK-NEXT:| |   `-CompoundStmt {{.*}} <col:17, line:217:5>
// CHECK-NEXT:| |     |-DeclStmt {{.*}} <line:215:9, col:30>
// CHECK-NEXT:| |     | `-VarDecl {{.*}} <col:9, col:26> col:13 referenced size_local 'int' cinit
// CHECK-NEXT:| |     |   `-ImplicitCastExpr {{.*}} <col:26> 'int' <LValueToRValue>
// CHECK-NEXT:| |     |     `-MemberExpr {{.*}} <col:26> 'int' lvalue ->size {{.*}}
// CHECK-NEXT:| |     |       `-CXXThisExpr {{.*}} <col:26> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| |     `-DeclStmt {{.*}} <line:216:9, col:44>
// CHECK-NEXT:| |       `-VarDecl {{.*}} <col:9, col:42> col:36 tmp 'T' cinit
// CHECK-NEXT:| |         |-MemberExpr {{.*}} <col:42> 'T' lvalue ->cb {{.*}}
// CHECK-NEXT:| |         | `-CXXThisExpr {{.*}} <col:42> 'RefParamMustBePtrExternallyCountedGood<T> *' implicit this
// CHECK-NEXT:| |         `-CountedByAttr {{.*}} <{{.+}}ptrcheck.h:56:40, col:56> 0
// CHECK-NEXT:| |           `-DeclRefExpr {{.*}} <{{.*}}attributes_in_template_decls_attr_only_mode.cpp:216:24> 'int' lvalue Var {{.*}} 'size_local' 'int'
// CHECK-NEXT:| |-ClassTemplateSpecialization {{.*}} 'RefParamMustBePtrExternallyCountedGood'
// CHECK-NEXT:| |-ClassTemplateSpecializationDecl {{.*}} <line:194:1, line:218:1> line:195:7 class RefParamMustBePtrExternallyCountedGood definition implicit_instantiation
// CHECK-NEXT:| | |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | | |-DefaultConstructor exists trivial
// CHECK-NEXT:| | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:| | | |-MoveConstructor exists simple trivial
// CHECK-NEXT:| | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| | |-TemplateArgument type 'int *'
// CHECK-NEXT:| | | `-PointerType {{.*}} 'int *'
// CHECK-NEXT:| | |   `-BuiltinType {{.*}} 'int'
// CHECK-NEXT:| | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrExternallyCountedGood
// CHECK-NEXT:| | |-AccessSpecDecl {{.*}} <line:196:5, col:11> col:5 public
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:197:5, col:9> col:9 referenced size 'int'
// CHECK-NEXT:| | | `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} {{.*}} {{.*}} {{.*}} 0 0 0 0
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:198:5, col:7> col:7 referenced end_ptr 'int * /* __started_by(eb) */ ':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:199:5, col:26> col:26 cb 'int * __counted_by(size)':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:200:5, col:34> col:34 cbon 'int * __counted_by_or_null(size)':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:201:5, col:24> col:24 sb 'int * __sized_by(size)':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:202:5, col:32> col:32 sbon 'int * __sized_by_or_null(size)':'int *'
// CHECK-NEXT:| | |-FieldDecl {{.*}} <line:203:5, col:27> col:27 referenced eb 'int * __ended_by(end_ptr)':'int *'
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:205:5, line:207:5> line:205:26 ret_cb 'int * __counted_by(size)()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXMethodDecl {{.*}} <line:214:5, line:217:5> line:214:10 useT 'void ()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | |-CXXConstructorDecl {{.*}} <line:195:7> col:7 implicit used RefParamMustBePtrExternallyCountedGood 'void () noexcept' inline default trivial
// CHECK-NEXT:| | | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:| | |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrExternallyCountedGood 'void (const RefParamMustBePtrExternallyCountedGood<int *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:| | | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamMustBePtrExternallyCountedGood<int *> &'
// CHECK-NEXT:| | `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrExternallyCountedGood 'void (RefParamMustBePtrExternallyCountedGood<int *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:| |   `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamMustBePtrExternallyCountedGood<int *> &&'
// CHECK-NEXT:| `-ClassTemplateSpecializationDecl {{.*}} <line:194:1, line:218:1> line:195:7 class RefParamMustBePtrExternallyCountedGood definition implicit_instantiation
// CHECK-NEXT:|   |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:|   | |-DefaultConstructor exists trivial
// CHECK-NEXT:|   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:|   | |-MoveConstructor exists simple trivial
// CHECK-NEXT:|   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:|   | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:|   | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:|   |-TemplateArgument type 'char *'
// CHECK-NEXT:|   | `-PointerType {{.*}} 'char *'
// CHECK-NEXT:|   |   `-BuiltinType {{.*}} 'char'
// CHECK-NEXT:|   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class RefParamMustBePtrExternallyCountedGood
// CHECK-NEXT:|   |-AccessSpecDecl {{.*}} <line:196:5, col:11> col:5 public
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:197:5, col:9> col:9 referenced size 'int'
// CHECK-NEXT:|   | `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} {{.*}} {{.*}} {{.*}} 0 0 0 0
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:198:5, col:7> col:7 referenced end_ptr 'char * /* __started_by(eb) */ ':'char *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:199:5, col:26> col:26 cb 'char * __counted_by(size)':'char *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:200:5, col:34> col:34 cbon 'char * __counted_by_or_null(size)':'char *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:201:5, col:24> col:24 sb 'char * __sized_by(size)':'char *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:202:5, col:32> col:32 sbon 'char * __sized_by_or_null(size)':'char *'
// CHECK-NEXT:|   |-FieldDecl {{.*}} <line:203:5, col:27> col:27 referenced eb 'char * __ended_by(end_ptr)':'char *'
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:205:5, line:207:5> line:205:26 ret_cb 'char * __counted_by(size)()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXMethodDecl {{.*}} <line:214:5, line:217:5> line:214:10 useT 'void ()' implicit_instantiation implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <line:195:7> col:7 implicit used RefParamMustBePtrExternallyCountedGood 'void () noexcept' inline default trivial
// CHECK-NEXT:|   | `-CompoundStmt {{.*}} <col:7>
// CHECK-NEXT:|   |-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrExternallyCountedGood 'void (const RefParamMustBePtrExternallyCountedGood<char *> &)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|   | `-ParmVarDecl {{.*}} <col:7> col:7 'const RefParamMustBePtrExternallyCountedGood<char *> &'
// CHECK-NEXT:|   `-CXXConstructorDecl {{.*}} <col:7> col:7 implicit constexpr RefParamMustBePtrExternallyCountedGood 'void (RefParamMustBePtrExternallyCountedGood<char *> &&)' inline default trivial noexcept-unevaluated {{.*}}
// CHECK-NEXT:|     `-ParmVarDecl {{.*}} <col:7> col:7 'RefParamMustBePtrExternallyCountedGood<char *> &&'
// CHECK-NEXT:|-ClassTemplateSpecializationDecl {{.*}} <line:221:1, col:61> col:16 class RefParamMustBePtrExternallyCountedGood definition explicit_instantiation_definition
// CHECK-NEXT:| |-DefinitionData pass_in_registers aggregate standard_layout trivially_copyable pod trivial literal
// CHECK-NEXT:| | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:| | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:| | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:| | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:| | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:| |-TemplateArgument type 'float *'
// CHECK-NEXT:| | `-PointerType {{.*}} 'float *'
// CHECK-NEXT:| |   `-BuiltinType {{.*}} 'float'
// CHECK-NEXT:| |-CXXRecordDecl {{.*}} <line:195:1, col:7> col:7 implicit class RefParamMustBePtrExternallyCountedGood
// CHECK-NEXT:| |-AccessSpecDecl {{.*}} <line:196:5, col:11> col:5 public
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:197:5, col:9> col:9 referenced size 'int'
// CHECK-NEXT:| | `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} {{.*}} {{.*}} {{.*}} 0 0 0 0
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:198:5, col:7> col:7 referenced end_ptr 'float * /* __started_by(eb) */ ':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:199:5, col:26> col:26 referenced cb 'float * __counted_by(size)':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:200:5, col:34> col:34 cbon 'float * __counted_by_or_null(size)':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:201:5, col:24> col:24 sb 'float * __sized_by(size)':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:202:5, col:32> col:32 sbon 'float * __sized_by_or_null(size)':'float *'
// CHECK-NEXT:| |-FieldDecl {{.*}} <line:203:5, col:27> col:27 referenced eb 'float * __ended_by(end_ptr)':'float *'
// CHECK-NEXT:| |-CXXMethodDecl {{.*}} <line:205:5, line:207:5> line:205:26 ret_cb 'float * __counted_by(size)()' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:| | `-CompoundStmt {{.*}} <col:35, line:207:5>
// CHECK-NEXT:| |   `-ReturnStmt {{.*}} <line:206:9, col:16>
// CHECK-NEXT:| |     `-ImplicitCastExpr {{.*}} <col:16> 'float * __counted_by(size)':'float *' <LValueToRValue>
// CHECK-NEXT:| |       `-MemberExpr {{.*}} <col:16> 'float * __counted_by(size)':'float *' lvalue ->cb {{.*}}
// CHECK-NEXT:| |         `-CXXThisExpr {{.*}} <col:16> 'RefParamMustBePtrExternallyCountedGood<float *> *' implicit this
// CHECK-NEXT:| `-CXXMethodDecl {{.*}} <line:214:5, line:217:5> line:214:10 useT 'void ()' explicit_instantiation_definition implicit-inline instantiated_from {{.*}}
// CHECK-NEXT:|   `-CompoundStmt {{.*}} <col:17, line:217:5>
// CHECK-NEXT:|     |-DeclStmt {{.*}} <line:215:9, col:30>
// CHECK-NEXT:|     | `-VarDecl {{.*}} <col:9, col:26> col:13 used size_local 'int' cinit
// CHECK-NEXT:|     |   |-ImplicitCastExpr {{.*}} <col:26> 'int' <LValueToRValue>
// CHECK-NEXT:|     |   | `-MemberExpr {{.*}} <col:26> 'int' lvalue ->size {{.*}}
// CHECK-NEXT:|     |   |   `-CXXThisExpr {{.*}} <col:26> 'RefParamMustBePtrExternallyCountedGood<float *> *' implicit this
// CHECK-NEXT:|     |   `-DependerDeclsAttr {{.*}} <<invalid sloc>> Implicit {{.*}} 0
// CHECK-NEXT:|     `-DeclStmt {{.*}} <line:216:9, col:44>
// CHECK-NEXT:|       `-VarDecl {{.*}} <col:9, col:42> col:36 tmp 'float * __counted_by(size_local)':'float *' cinit
// CHECK-NEXT:|         `-ImplicitCastExpr {{.*}} <col:42> 'float * __counted_by(size)':'float *' <LValueToRValue>
// CHECK-NEXT:|           `-MemberExpr {{.*}} <col:42> 'float * __counted_by(size)':'float *' lvalue ->cb {{.*}}
// CHECK-NEXT:|             `-CXXThisExpr {{.*}} <col:42> 'RefParamMustBePtrExternallyCountedGood<float *> *' implicit this
// CHECK-NEXT:`-FunctionDecl {{.*}} <line:223:1, line:227:1> line:223:6 Instantiate_RefParamMustBePtrExternallyCountedGood 'void ()'
// CHECK-NEXT:  `-CompoundStmt {{.*}} <col:59, line:227:1>
// CHECK-NEXT:    |-DeclStmt {{.*}} <line:225:5, col:55>
// CHECK-NEXT:    | `-VarDecl {{.*}} <col:5, col:50> col:50 good0 'RefParamMustBePtrExternallyCountedGood<int *>' callinit
// CHECK-NEXT:    |   `-CXXConstructExpr {{.*}} <col:50> 'RefParamMustBePtrExternallyCountedGood<int *>' 'void () noexcept'
// CHECK-NEXT:    `-DeclStmt {{.*}} <line:226:5, col:61>
// CHECK-NEXT:      `-VarDecl {{.*}} <col:5, col:56> col:56 good1 'RefParamMustBePtrExternallyCountedGood<PtrTypedef>':'RefParamMustBePtrExternallyCountedGood<char *>' callinit
// CHECK-NEXT:        `-CXXConstructExpr {{.*}} <col:56> 'RefParamMustBePtrExternallyCountedGood<PtrTypedef>':'RefParamMustBePtrExternallyCountedGood<char *>' 'void () noexcept'
