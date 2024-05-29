// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify -fcxx-exceptions %s
// RUN: %clang_cc1 -std=c++98 -fsyntax-only -verify -fcxx-exceptions %s -fexperimental-new-constant-interpreter

//
// Tests for "expression traits" intrinsics such as __is_lvalue_expr.
//
// For the time being, these tests are written against the 2003 C++
// standard (ISO/IEC 14882:2003 -- see draft at
// http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2001/n1316/).
//
// C++0x has its own, more-refined, idea of lvalues and rvalues.
// If/when we need to support those, we'll need to track both
// standard documents.

#if !__has_feature(cxx_static_assert)
# define CONCAT_(X_, Y_) CONCAT1_(X_, Y_)
# define CONCAT1_(X_, Y_) X_ ## Y_

// This emulation can be used multiple times on one line (and thus in
// a macro), except at class scope
# define static_assert(b_, m_) \
  typedef int CONCAT_(sa_, __LINE__)[b_ ? 1 : -1]
#endif

// Tests are broken down according to section of the C++03 standard
// (ISO/IEC 14882:2003(E))

// Assertion macros encoding the following two paragraphs
//
// basic.lval/1 Every expression is either an lvalue or an rvalue.
//
// expr.prim/5 A parenthesized expression is a primary expression whose type
// and value are identical to those of the enclosed expression. The
// presence of parentheses does not affect whether the expression is
// an lvalue.
//
// Note: these asserts cannot be made at class scope in C++03.  Put
// them in a member function instead.
#define ASSERT_LVALUE(expr)                                             \
    static_assert(__is_lvalue_expr(expr), "should be an lvalue");       \
    static_assert(__is_lvalue_expr((expr)),                             \
                  "the presence of parentheses should have"             \
                  " no effect on lvalueness (expr.prim/5)");            \
    static_assert(!__is_rvalue_expr(expr), "should be an lvalue");      \
    static_assert(!__is_rvalue_expr((expr)),                            \
                  "the presence of parentheses should have"             \
                  " no effect on lvalueness (expr.prim/5)")

#define ASSERT_RVALUE(expr);                                            \
    static_assert(__is_rvalue_expr(expr), "should be an rvalue");       \
    static_assert(__is_rvalue_expr((expr)),                             \
                  "the presence of parentheses should have"             \
                  " no effect on lvalueness (expr.prim/5)");            \
    static_assert(!__is_lvalue_expr(expr), "should be an rvalue");      \
    static_assert(!__is_lvalue_expr((expr)),                            \
                  "the presence of parentheses should have"             \
                  " no effect on lvalueness (expr.prim/5)")

enum Enum { Enumerator };

int ReturnInt();
void ReturnVoid();
Enum ReturnEnum();

void basic_lval_5()
{
    // basic.lval/5: The result of calling a function that does not return
    // a reference is an rvalue.
    ASSERT_RVALUE(ReturnInt());
    ASSERT_RVALUE(ReturnVoid());
    ASSERT_RVALUE(ReturnEnum());
}

int& ReturnIntReference();
extern Enum& ReturnEnumReference();

void basic_lval_6()
{
    // basic.lval/6: An expression which holds a temporary object resulting
    // from a cast to a nonreference type is an rvalue (this includes
    // the explicit creation of an object using functional notation
    struct IntClass
    {
        explicit IntClass(int = 0);
        IntClass(char const*);
        operator int() const;
    };
    
    struct ConvertibleToIntClass
    {
        operator IntClass() const;
    };

    ConvertibleToIntClass b;

    // Make sure even trivial conversions are not detected as lvalues
    int intLvalue = 0;
    ASSERT_RVALUE((int)intLvalue);
    ASSERT_RVALUE((short)intLvalue);
    ASSERT_RVALUE((long)intLvalue);
    
    // Same tests with function-call notation
    ASSERT_RVALUE(int(intLvalue));
    ASSERT_RVALUE(short(intLvalue));
    ASSERT_RVALUE(long(intLvalue));

    char charLValue = 'x';
    ASSERT_RVALUE((signed char)charLValue);
    ASSERT_RVALUE((unsigned char)charLValue);

    ASSERT_RVALUE(static_cast<int>(IntClass()));
    IntClass intClassLValue;
    ASSERT_RVALUE(static_cast<int>(intClassLValue)); 
    ASSERT_RVALUE(static_cast<IntClass>(ConvertibleToIntClass()));
    ConvertibleToIntClass convertibleToIntClassLValue;
    ASSERT_RVALUE(static_cast<IntClass>(convertibleToIntClassLValue));
    

    typedef signed char signed_char;
    typedef unsigned char unsigned_char;
    ASSERT_RVALUE(signed_char(charLValue));
    ASSERT_RVALUE(unsigned_char(charLValue));

    ASSERT_RVALUE(int(IntClass()));
    ASSERT_RVALUE(int(intClassLValue)); 
    ASSERT_RVALUE(IntClass(ConvertibleToIntClass()));
    ASSERT_RVALUE(IntClass(convertibleToIntClassLValue));
}

void conv_ptr_1()
{
    // conv.ptr/1: A null pointer constant is an integral constant
    // expression (5.19) rvalue of integer type that evaluates to
    // zero.
    ASSERT_RVALUE(0);
}

void expr_6()
{
    // expr/6: If an expression initially has the type "reference to T"
    // (8.3.2, 8.5.3), ... the expression is an lvalue.
    int x = 0;
    int& referenceToInt = x;
    ASSERT_LVALUE(referenceToInt);
    ASSERT_LVALUE(ReturnIntReference());
}

void expr_prim_2()
{
    // 5.1/2 A string literal is an lvalue; all other
    // literals are rvalues.
    ASSERT_LVALUE("foo");
    ASSERT_RVALUE(1);
    ASSERT_RVALUE(1.2);
    ASSERT_RVALUE(10UL);
}

void expr_prim_3()
{
    // 5.1/3: The keyword "this" names a pointer to the object for
    // which a nonstatic member function (9.3.2) is invoked. ...The
    // expression is an rvalue.
    struct ThisTest
    {
        void f() { ASSERT_RVALUE(this); }
    };
}

extern int variable;
void Function();

struct BaseClass
{
    virtual ~BaseClass();
    
    int BaseNonstaticMemberFunction();
    static int BaseStaticMemberFunction();
    int baseDataMember;
};

struct Class : BaseClass
{
    static void function();
    static int variable;

    template <class T>
    struct NestedClassTemplate {};

    template <class T>
    static int& NestedFuncTemplate() { return variable; }  // expected-note{{possible target for call}}

    template <class T>
    int& NestedMemfunTemplate() { return variable; } // expected-note{{possible target for call}}

    int operator*() const;

    template <class T>
    int operator+(T) const; // expected-note{{possible target for call}}

    int NonstaticMemberFunction();
    static int StaticMemberFunction();
    int dataMember;

    int& referenceDataMember;
    static int& staticReferenceDataMember;
    static int staticNonreferenceDataMember;

    enum Enum { Enumerator };

    operator long() const;
    
    Class();
    Class(int,int);

    void expr_prim_4()
    {
        // 5.1/4: The operator :: followed by an identifier, a
        // qualified-id, or an operator-function-id is a primary-
        // expression. ...The result is an lvalue if the entity is
        // a function or variable.
        ASSERT_LVALUE(::Function);         // identifier: function
        ASSERT_LVALUE(::variable);         // identifier: variable

        // the only qualified-id form that can start without "::" (and thus
        // be legal after "::" ) is
        //
        // ::<sub>opt</sub> nested-name-specifier template<sub>opt</sub> unqualified-id
        ASSERT_LVALUE(::Class::function);  // qualified-id: function
        ASSERT_LVALUE(::Class::variable);  // qualified-id: variable

        // The standard doesn't give a clear answer about whether these
        // should really be lvalues or rvalues without some surrounding
        // context that forces them to be interpreted as naming a
        // particular function template specialization (that situation
        // doesn't come up in legal pure C++ programs). This language
        // extension simply rejects them as requiring additional context
        __is_lvalue_expr(::Class::NestedFuncTemplate);    // qualified-id: template \
        // expected-error{{reference to overloaded function could not be resolved; did you mean to call it?}}
        
        __is_lvalue_expr(::Class::NestedMemfunTemplate);  // qualified-id: template \
        // expected-error{{reference to non-static member function must be called}}
        
        __is_lvalue_expr(::Class::operator+);             // operator-function-id: template \
        // expected-error{{reference to non-static member function must be called}}

        //ASSERT_RVALUE(::Class::operator*);         // operator-function-id: member function
    }

    void expr_prim_7()
    {
        // expr.prim/7 An identifier is an id-expression provided it has been
        // suitably declared (clause 7). [Note: ... ] The type of the
        // expression is the type of the identifier. The result is the
        // entity denoted by the identifier. The result is an lvalue if
        // the entity is a function, variable, or data member... (cont'd)
        ASSERT_LVALUE(Function);        // identifier: function
        ASSERT_LVALUE(StaticMemberFunction);        // identifier: function
        ASSERT_LVALUE(variable);        // identifier: variable
        ASSERT_LVALUE(dataMember);      // identifier: data member
        //ASSERT_RVALUE(NonstaticMemberFunction); // identifier: member function

        // (cont'd)...A nested-name-specifier that names a class,
        // optionally followed by the keyword template (14.2), and then
        // followed by the name of a member of either that class (9.2) or
        // one of its base classes... is a qualified-id... The result is
        // the member. The type of the result is the type of the
        // member. The result is an lvalue if the member is a static
        // member function or a data member.
        ASSERT_LVALUE(Class::dataMember);
        ASSERT_LVALUE(Class::StaticMemberFunction);
        //ASSERT_RVALUE(Class::NonstaticMemberFunction); // identifier: member function

        ASSERT_LVALUE(Class::baseDataMember);
        ASSERT_LVALUE(Class::BaseStaticMemberFunction);
        //ASSERT_RVALUE(Class::BaseNonstaticMemberFunction); // identifier: member function
    }
};

void expr_call_10()
{
    // expr.call/10: A function call is an lvalue if and only if the
    // result type is a reference.  This statement is partially
    // redundant with basic.lval/5
    basic_lval_5();
    
    ASSERT_LVALUE(ReturnIntReference());
    ASSERT_LVALUE(ReturnEnumReference());
}

namespace Namespace
{
  int x;
  void function();
}

void expr_prim_8()
{
    // expr.prim/8 A nested-name-specifier that names a namespace
    // (7.3), followed by the name of a member of that namespace (or
    // the name of a member of a namespace made visible by a
    // using-directive ) is a qualified-id; 3.4.3.2 describes name
    // lookup for namespace members that appear in qualified-ids. The
    // result is the member. The type of the result is the type of the
    // member. The result is an lvalue if the member is a function or
    // a variable.
    ASSERT_LVALUE(Namespace::x);
    ASSERT_LVALUE(Namespace::function);
}

void expr_sub_1(int* pointer)
{
    // expr.sub/1 A postfix expression followed by an expression in
    // square brackets is a postfix expression. One of the expressions
    // shall have the type "pointer to T" and the other shall have
    // enumeration or integral type. The result is an lvalue of type
    // "T."
    ASSERT_LVALUE(pointer[1]);
    
    // The expression E1[E2] is identical (by definition) to *((E1)+(E2)).
    ASSERT_LVALUE(*(pointer+1));
}

void expr_type_conv_1()
{
    // expr.type.conv/1 A simple-type-specifier (7.1.5) followed by a
    // parenthesized expression-list constructs a value of the specified
    // type given the expression list. ... If the expression list
    // specifies more than a single value, the type shall be a class with
    // a suitably declared constructor (8.5, 12.1), and the expression
    // T(x1, x2, ...) is equivalent in effect to the declaration T t(x1,
    // x2, ...); for some invented temporary variable t, with the result
    // being the value of t as an rvalue.
    ASSERT_RVALUE(Class(2,2));
}

void expr_type_conv_2()
{
    // expr.type.conv/2 The expression T(), where T is a
    // simple-type-specifier (7.1.5.2) for a non-array complete object
    // type or the (possibly cv-qualified) void type, creates an
    // rvalue of the specified type,
    ASSERT_RVALUE(int());
    ASSERT_RVALUE(Class());
    ASSERT_RVALUE(void());
}


void expr_ref_4()
{
    // Applies to expressions of the form E1.E2
    
    // If E2 is declared to have type "reference to T", then E1.E2 is
    // an lvalue;.... Otherwise, one of the following rules applies.
    ASSERT_LVALUE(Class().staticReferenceDataMember);
    ASSERT_LVALUE(Class().referenceDataMember);
    
    // - If E2 is a static data member, and the type of E2 is T, then
    // E1.E2 is an lvalue; ...
    ASSERT_LVALUE(Class().staticNonreferenceDataMember);
    ASSERT_LVALUE(Class().staticReferenceDataMember);


    // - If E2 is a non-static data member, ... If E1 is an lvalue,
    // then E1.E2 is an lvalue...
    Class lvalue;
    ASSERT_LVALUE(lvalue.dataMember);
    ASSERT_RVALUE(Class().dataMember);

    // - If E1.E2 refers to a static member function, ... then E1.E2
    // is an lvalue
    ASSERT_LVALUE(Class().StaticMemberFunction);
    
    // - Otherwise, if E1.E2 refers to a non-static member function,
    // then E1.E2 is not an lvalue.
    //ASSERT_RVALUE(Class().NonstaticMemberFunction);

    // - If E2 is a member enumerator, and the type of E2 is T, the
    // expression E1.E2 is not an lvalue. The type of E1.E2 is T.
    ASSERT_RVALUE(Class().Enumerator);
    ASSERT_RVALUE(lvalue.Enumerator);
}


void expr_post_incr_1(int x)
{
    // expr.post.incr/1 The value obtained by applying a postfix ++ is
    // the value that the operand had before applying the
    // operator... The result is an rvalue.
    ASSERT_RVALUE(x++);
}

void expr_dynamic_cast_2()
{
    // expr.dynamic.cast/2: If T is a pointer type, v shall be an
    // rvalue of a pointer to complete class type, and the result is
    // an rvalue of type T.
    Class instance;
    ASSERT_RVALUE(dynamic_cast<Class*>(&instance));

    // If T is a reference type, v shall be an
    // lvalue of a complete class type, and the result is an lvalue of
    // the type referred to by T.
    ASSERT_LVALUE(dynamic_cast<Class&>(instance));
}

void expr_dynamic_cast_5()
{
    // expr.dynamic.cast/5: If T is "reference to cv1 B" and v has type
    // "cv2 D" such that B is a base class of D, the result is an
    // lvalue for the unique B sub-object of the D object referred
    // to by v.
    typedef BaseClass B;
    typedef Class D;
    D object;
    ASSERT_LVALUE(dynamic_cast<B&>(object));
}

// expr.dynamic.cast/8: The run-time check logically executes as follows:
//
// - If, in the most derived object pointed (referred) to by v, v
// points (refers) to a public base class subobject of a T object, and
// if only one object of type T is derived from the sub-object pointed
// (referred) to by v, the result is a pointer (an lvalue referring)
// to that T object.
//
// - Otherwise, if v points (refers) to a public base class sub-object
// of the most derived object, and the type of the most derived object
// has a base class, of type T, that is unambiguous and public, the
// result is a pointer (an lvalue referring) to the T sub-object of
// the most derived object.
//
// The mention of "lvalue" in the text above appears to be a
// defect that is being corrected by the response to UK65 (see
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2009/n2841.html).

#if 0
void expr_typeid_1()
{
    // expr.typeid/1: The result of a typeid expression is an lvalue...
    ASSERT_LVALUE(typeid(1));
}
#endif

void expr_static_cast_1(int x)
{
    // expr.static.cast/1: The result of the expression
    // static_cast<T>(v) is the result of converting the expression v
    // to type T. If T is a reference type, the result is an lvalue;
    // otherwise, the result is an rvalue.
    ASSERT_LVALUE(static_cast<int&>(x));
    ASSERT_RVALUE(static_cast<int>(x));
}

void expr_reinterpret_cast_1()
{
    // expr.reinterpret.cast/1: The result of the expression
    // reinterpret_cast<T>(v) is the result of converting the
    // expression v to type T. If T is a reference type, the result is
    // an lvalue; otherwise, the result is an rvalue
    ASSERT_RVALUE(reinterpret_cast<int*>(0));
    char const v = 0;
    ASSERT_LVALUE(reinterpret_cast<char const&>(v));
}

void expr_unary_op_1(int* pointer, struct incomplete* pointerToIncompleteType)
{
    // expr.unary.op/1: The unary * operator performs indirection: the
    // expression to which it is applied shall be a pointer to an
    // object type, or a pointer to a function type and the result is
    // an lvalue referring to the object or function to which the
    // expression points.  
    ASSERT_LVALUE(*pointer);
    ASSERT_LVALUE(*Function);

    // [Note: a pointer to an incomplete type
    // (other than cv void ) can be dereferenced. ]
    ASSERT_LVALUE(*pointerToIncompleteType);
}

void expr_pre_incr_1(int operand)
{
    // expr.pre.incr/1: The operand of prefix ++ ... shall be a
    // modifiable lvalue.... The value is the new value of the
    // operand; it is an lvalue.
    ASSERT_LVALUE(++operand);
}

void expr_cast_1(int x)
{
    // expr.cast/1: The result of the expression (T) cast-expression
    // is of type T. The result is an lvalue if T is a reference type,
    // otherwise the result is an rvalue.
    ASSERT_LVALUE((void(&)())expr_cast_1);
    ASSERT_LVALUE((int&)x);
    ASSERT_RVALUE((void(*)())expr_cast_1);
    ASSERT_RVALUE((int)x);
}

void expr_mptr_oper()
{
    // expr.mptr.oper/6: The result of a .* expression is an lvalue
    // only if its first operand is an lvalue and its second operand
    // is a pointer to data member... (cont'd)
    typedef Class MakeRValue;
    ASSERT_RVALUE(MakeRValue().*(&Class::dataMember));
    //ASSERT_RVALUE(MakeRValue().*(&Class::NonstaticMemberFunction));
    Class lvalue;
    ASSERT_LVALUE(lvalue.*(&Class::dataMember));
    //ASSERT_RVALUE(lvalue.*(&Class::NonstaticMemberFunction));
    
    // (cont'd)...The result of an ->* expression is an lvalue only
    // if its second operand is a pointer to data member. If the
    // second operand is the null pointer to member value (4.11), the
    // behavior is undefined.
    ASSERT_LVALUE((&lvalue)->*(&Class::dataMember));
    //ASSERT_RVALUE((&lvalue)->*(&Class::NonstaticMemberFunction));
}

void expr_cond(bool cond)
{
    // 5.16 Conditional operator [expr.cond]
    //
    // 2 If either the second or the third operand has type (possibly
    // cv-qualified) void, one of the following shall hold:
    //
    // - The second or the third operand (but not both) is a
    // (possibly parenthesized) throw-expression (15.1); the result
    // is of the type and value category of the other.

    Class classLvalue;
    ASSERT_RVALUE(cond ? throw 1 : (void)0);
    ASSERT_RVALUE(cond ? (void)0 : throw 1);
    ASSERT_RVALUE(cond ? throw 1 : 0);
    ASSERT_RVALUE(cond ? 0 : throw 1);
    ASSERT_LVALUE(cond ? throw 1 : classLvalue);
    ASSERT_LVALUE(cond ? classLvalue : throw 1);

    // - Both the second and the third operands have type void; the result
    // is of type void and is an rvalue. [Note: this includes the case
    // where both operands are throw-expressions. ]
    ASSERT_RVALUE(cond ? (void)1 : (void)0);
    ASSERT_RVALUE(cond ? throw 1 : throw 0);
    
    // expr.cond/4: If the second and third operands are lvalues and
    // have the same type, the result is of that type and is an
    // lvalue.
    ASSERT_LVALUE(cond ? classLvalue : classLvalue);
    int intLvalue = 0;
    ASSERT_LVALUE(cond ? intLvalue : intLvalue);
    
    // expr.cond/5:Otherwise, the result is an rvalue.
    typedef Class MakeRValue;
    ASSERT_RVALUE(cond ? MakeRValue() : classLvalue);
    ASSERT_RVALUE(cond ? classLvalue : MakeRValue());
    ASSERT_RVALUE(cond ? MakeRValue() : MakeRValue());
    ASSERT_RVALUE(cond ? classLvalue : intLvalue);
    ASSERT_RVALUE(cond ? intLvalue : int());
}

void expr_ass_1(int x)
{
    // expr.ass/1: There are several assignment operators, all of
    // which group right-to-left. All require a modifiable lvalue as
    // their left operand, and the type of an assignment expression is
    // that of its left operand. The result of the assignment
    // operation is the value stored in the left operand after the
    // assignment has taken place; the result is an lvalue.
    ASSERT_LVALUE(x = 1);
    ASSERT_LVALUE(x += 1);
    ASSERT_LVALUE(x -= 1);
    ASSERT_LVALUE(x *= 1);
    ASSERT_LVALUE(x /= 1);
    ASSERT_LVALUE(x %= 1);
    ASSERT_LVALUE(x ^= 1);
    ASSERT_LVALUE(x &= 1);
    ASSERT_LVALUE(x |= 1);
}

void expr_comma(int x)
{
    // expr.comma: A pair of expressions separated by a comma is
    // evaluated left-to-right and the value of the left expression is
    // discarded... result is an lvalue if its right operand is.

    // Can't use the ASSERT_XXXX macros without adding parens around
    // the comma expression.
    static_assert(__is_lvalue_expr((void)x,x), "expected an lvalue");
    static_assert(__is_rvalue_expr((void)x,1), "expected an rvalue");
    static_assert(__is_lvalue_expr((void)1,x), "expected an lvalue");
    static_assert(__is_rvalue_expr((void)1,1), "expected an rvalue");
}

#if 0
template<typename T> void f();

// FIXME These currently fail
void expr_fun_lvalue()
{
  ASSERT_LVALUE(&f<int>);
}

void expr_fun_rvalue()
{
  ASSERT_RVALUE(f<int>);
}
#endif

template <int NonTypeNonReferenceParameter, int& NonTypeReferenceParameter>
void check_temp_param_6()
{
    ASSERT_RVALUE(NonTypeNonReferenceParameter);
    ASSERT_LVALUE(NonTypeReferenceParameter);
}

int AnInt = 0;

void temp_param_6()
{
    check_temp_param_6<3,AnInt>();
}
