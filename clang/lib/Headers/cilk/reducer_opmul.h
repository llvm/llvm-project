/*  reducer_opmul.h                  -*- C++ -*-
 *
 *  @copyright
 *  Copyright (C) 2012-2013, Intel Corporation
 *  All rights reserved.
 *  
 *  @copyright
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in
 *      the documentation and/or other materials provided with the
 *      distribution.
 *    * Neither the name of Intel Corporation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *  
 *  @copyright
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 *  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 *  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/** @file reducer_opmul.h
 *
 *  @brief Defines classes for doing parallel multiplication reductions.
 *
 *  @ingroup ReducersMul
 *
 *  @see ReducersMul
 */

#ifndef REDUCER_OPMUL_H_INCLUDED
#define REDUCER_OPMUL_H_INCLUDED

#include <cilk/reducer.h>

/** @defgroup ReducersMul Multiplication Reducers
 *
 *  Multiplication reducers allow the computation of the product of a set of
 *  values in parallel.
 *
 *  @ingroup Reducers
 *
 *  You should be familiar with @ref pagereducers "Cilk reducers", described in
 *  file `reducers.md`, and particularly with @ref reducers_using, before trying
 *  to use the information in this file.
 *
 *  @section redopmul_usage Usage Example
 *
 *      cilk::reducer< cilk::op_mul<double> > r;
 *      cilk_for (int i = 0; i != N; ++i) {
 *          *r *= a[i];
 *      }
 *      double product;
 *      r.move_out(product);
 *
 *  @section redopmul_monoid The Monoid
 *
 *  @subsection redopmul_monoid_values Value Set
 *
 *  The value set of a multiplication reducer is the set of values of `Type`,
 *  which is expected to be a builtin numeric type (or something like it, such
 *  as `std::complex`).
 *
 *  @subsection redopmul_monoid_operator Operator
 *
 *  The operator of a multiplication reducer is the multiplication operation,
 *  defined by the “`*`” binary operator on `Type`.
 *
 *  @subsection redopmul_monoid_identity Identity
 *
 *  The identity value of the reducer is the numeric value “`1`”. This is
 *  expected to be the value of the expression `Type(1)`.
 *
 *  @section redopmul_operations Operations
 *
 *  @subsection redopmul_constructors Constructors
 *
 *      reducer()   // identity
 *      reducer(const Type& value)
 *      reducer(move_in(Type& variable))
 *
 *  @subsection redopmul_get_set Set and Get
 *
 *      r.set_value(const Type& value)
 *      const Type& = r.get_value() const
 *      r.move_in(Type& variable)
 *      r.move_out(Type& variable)
 *
 *  @subsection redopmul_initial Initial Values
 *
 *  If a multiplication reducer is constructed without an explicit initial
 *  value, then its initial value will be its identity value, as long as `Type`
 *  satisfies the requirements of @ref redopmul_types.
 *
 *  @subsection redopmul_view_ops View Operations
 *
 *      *r *= a
 *      *r = *r * a
 *      *r = *r * a1 * a2 … * an
 *
 *  @section redopmul_floating_point Issues with Floating-Point Types
 *
 *  Because of overflow and underflow issues, floating-point multiplication is
 *  not really associative. For example, `(1e200 * 1e-200) * 1e-200 == 1e-200`,
 *  but `1e200 * (1e-200 * 1e-200 == 0.
 *
 *  In many cases, this won’t matter, but computations which have been
 *  carefully ordered to control overflow and underflow may not deal well with
 *  being reassociated. In general, you should be sure to understand the
 *  floating-point behavior of your program before doing any transformation 
 *  that will reassociate its computations. 
 *
 *  @section redopmul_types Type and Operator Requirements
 *
 *  `Type` must be `Copy Constructible`, `Default Constructible`, and 
 *  `Assignable`.
 *
 *  The operator “`*=`” must be defined on `Type`, with `x *= a` having the same
 *  meaning as `x = x * a`.
 *
 *  The expression `Type(1)` must be a valid expression which yields the
 *  identity value (the value of `Type` whose numeric value is `1`).
 *
 *  @section redopmul_in_c Multiplication Reducers in C
 *
 *  The @ref CILK_C_REDUCER_OPMUL and @ref CILK_C_REDUCER_OPMUL_TYPE macros can
 *  be used to do multiplication reductions in C. For example:
 *
 *      CILK_C_REDUCER_OPMUL(r, double, 1);
 *      CILK_C_REGISTER_REDUCER(r);
 *      cilk_for(int i = 0; i != n; ++i) {
 *          REDUCER_VIEW(r) *= a[i];
 *      }
 *      CILK_C_UNREGISTER_REDUCER(r);
 *      printf("The product of the elements of a is %f\n", REDUCER_VIEW(r));
 *
 *  See @ref reducers_c_predefined.
 */

#ifdef __cplusplus

namespace cilk {

/** The multiplication reducer view class.
 *
 *  This is the view class for reducers created with 
 *  `cilk::reducer< cilk::op_mul<Type> >`. It holds the accumulator variable 
 *  for the reduction, and allows only multiplication operations to be 
 *  performed on it.
 *
 *  @note   The reducer “dereference” operation (`reducer::operator *()`) 
 *          yields a reference to the view. Thus, for example, the view class’s
 *          `*=` operation would be used in an expression like `*r *= a`, where
 *          `r` is an op_mul reducer variable.
 *
 *  @tparam Type    The type of the contained accumulator variable. This will 
 *                  be the value type of a monoid_with_view that is 
 *                  instantiated with this view.
 *
 *  @see ReducersMul
 *  @see op_mul
 *
 *  @ingroup ReducersMul
 */
template <typename Type>
class op_mul_view : public scalar_view<Type>
{
    typedef scalar_view<Type> base;
    
public:
    /** Class to represent the right-hand side of `*reducer = *reducer * value`.
     *
     *  The only assignment operator for the op_mul_view class takes an 
     *  rhs_proxy as its operand. This results in the syntactic restriction 
     *  that the only expressions that can be assigned to an op_mul_view are
     *  ones which generate an rhs_proxy — that is, expressions of the form
     *  `op_mul_view * value ... * value`.
     *
     *  @warning
     *  The lhs and rhs views in such an assignment must be the same; 
     *  otherwise, the behavior will be undefined. (I.e., `v1 = v1 * x` is
     *  legal; `v1 = v2 * x` is illegal.) This condition will be checked with a
     *  runtime assertion when compiled in debug mode.
     *
     *  @see op_mul_view
     */
    class rhs_proxy {
        friend class op_mul_view;

        const op_mul_view* m_view;
        Type               m_value;

        // Constructor is invoked only from op_mul_view::operator*().
        //
        rhs_proxy(const op_mul_view* view, const Type& value) : m_view(view), m_value(value) {}

        rhs_proxy& operator=(const rhs_proxy&); // Disable assignment operator
        rhs_proxy();                            // Disable default constructor

    public:
        /** Multiply by an additional rhs value. If `v` is an op_mul_view and 
         *  `a1` is a value, then the expression `v * a1` invokes the view’s
         *  `operator*()` to create an rhs_proxy for `(v, a1)`; then 
         *  `v * a1 * a2` invokes the rhs_proxy’s `operator*()` to create a
         *  new rhs_proxy for `(v, a1*a2)`. This allows the right-hand side of
         *  an assignment to be not just `view * value`, but 
         *  `view * value * value ... * value`. The effect is that
         *
         *      v = v * a1 * a2 ... * an;
         *
         *  is evaluated as
         *
         *      v = v * (a1 * a2 ... * an);
         */
        rhs_proxy& operator*(const Type& x) { m_value *= x; return *this; }
    };


    /** Default/identity constructor. This constructor initializes the 
     *  contained value to `Type(1)`, which is expected to be the identity
     *  value for multiplication on `Type`.
     */
    op_mul_view() : base(Type(1)) {}

    /** Construct with a specified initial value.
     */
    explicit op_mul_view(const Type& v) : base(v) {}
    
    /** Reduction operation.
     *
     *  This function is invoked by the @ref op_mul monoid to combine the views
     *  of two strands when the right strand merges with the left one. It
     *  multiplies the value contained in the left-strand view by the value
     *  contained in the right-strand view, and leaves the value in the
     *  right-strand view undefined.
     *
     *  @param  right   A pointer to the right-strand view. (`this` points to
     *                  the left-strand view.)
     *
     *  @note   Used only by the @ref op_mul monoid to implement the monoid
     *          reduce operation.
     */
    void reduce(op_mul_view* right) { this->m_value *= right->m_value; }
    
    /** @name Accumulator variable updates.
     *
     *  These functions support the various syntaxes for multiplying the
     *  accumulator variable contained in the view by some value.
     */
    //@{

    /** Multiply the accumulator variable by @a x.
     */
    op_mul_view& operator*=(const Type& x) { this->m_value *= x; return *this; }

    /** Create an object representing `*this * x`.
     *
     *  @see rhs_proxy
     */
    rhs_proxy operator*(const Type& x) const { return rhs_proxy(this, x); }

    /** Assign the result of a `view * value` expression to the view. Note that
     *  this is the only assignment operator for this class.
     *
     *  @see rhs_proxy
     */
    op_mul_view& operator=(const rhs_proxy& rhs) {
        __CILKRTS_ASSERT(this == rhs.m_view);
        this->m_value *= rhs.m_value;
        return *this;
    }
    
    //@}
};

/** Monoid class for multiplication reductions. Instantiate the cilk::reducer
 *  template class with an op_mul monoid to create a multiplication reducer
 *  class. For example, to compute the product of a set of `double` values:
 *
 *      cilk::reducer< cilk::op_mul<double> > r;
 *
 *  @see ReducersMul
 *  @see op_mul_view
 *
 *  @ingroup ReducersMul
 */
template <typename Type>
struct op_mul : public monoid_with_view< op_mul_view<Type> > {};

} // namespace cilk

#endif // __cplusplus


/** @ingroup ReducersAdd
 */
//@{

/** @name C language reducer macros
 *
 *  These macros are used to declare and work with numeric op_mul reducers in
 *  C code.
 *
 *  @see @ref page_reducers_in_c
 */
 //@{
 
__CILKRTS_BEGIN_EXTERN_C

/** Opmul reducer type name.
 *
 *  This macro expands into the identifier which is the name of the op_mul
 *  reducer type for a specified numeric type.
 *
 *  @param  tn  The @ref reducers_c_type_names "numeric type name" specifying
 *              the type of the reducer.
 *
 *  @see @ref reducers_c_predefined
 *  @see ReducersMul
 */
#define CILK_C_REDUCER_OPMUL_TYPE(tn)                                         \
    __CILKRTS_MKIDENT(cilk_c_reducer_opmul_,tn)

/** Declare an op_mul reducer object.
 *
 *  This macro expands into a declaration of an op_mul reducer object for a
 *  specified numeric type. For example:
 *
 *      CILK_C_REDUCER_OPMUL(my_reducer, double, 1.0);
 *
 *  @param  obj The variable name to be used for the declared reducer object.
 *  @param  tn  The @ref reducers_c_type_names "numeric type name" specifying
 *              the type of the reducer.
 *  @param  v   The initial value for the reducer. (A value which can be
 *              assigned to the numeric type represented by @a tn.)
 *
 *  @see @ref reducers_c_predefined
 *  @see ReducersMul
 */
#define CILK_C_REDUCER_OPMUL(obj,tn,v)                                        \
    CILK_C_REDUCER_OPMUL_TYPE(tn) obj =                                       \
        CILK_C_INIT_REDUCER(_Typeof(obj.value),                               \
                        __CILKRTS_MKIDENT(cilk_c_reducer_opmul_reduce_,tn),   \
                        __CILKRTS_MKIDENT(cilk_c_reducer_opmul_identity_,tn), \
                        __cilkrts_hyperobject_noop_destroy, v)

/// @cond internal

/** Declare the op_mul reducer functions for a numeric type.
 *
 *  This macro expands into external function declarations for functions which 
 *  implement the reducer functionality for the op_mul reducer type for a
 *  specified numeric type.
 *
 *  @param  t   The value type of the reducer.
 *  @param  tn  The value “type name” identifier, used to construct the reducer
 *              type name, function names, etc.
 */
#define CILK_C_REDUCER_OPMUL_DECLARATION(t,tn)                             \
    typedef CILK_C_DECLARE_REDUCER(t) CILK_C_REDUCER_OPMUL_TYPE(tn);       \
    __CILKRTS_DECLARE_REDUCER_REDUCE(cilk_c_reducer_opmul,tn,l,r);         \
    __CILKRTS_DECLARE_REDUCER_IDENTITY(cilk_c_reducer_opmul,tn);
 
/** Define the op_mul reducer functions for a numeric type.
 *
 *  This macro expands into function definitions for functions which implement
 *  the reducer functionality for the op_mul reducer type for a specified
 *  numeric type.
 *
 *  @param  t   The value type of the reducer.
 *  @param  tn  The value “type name” identifier, used to construct the reducer
 *              type name, function names, etc.
 */
#define CILK_C_REDUCER_OPMUL_DEFINITION(t,tn)                              \
    typedef CILK_C_DECLARE_REDUCER(t) CILK_C_REDUCER_OPMUL_TYPE(tn);       \
    __CILKRTS_DECLARE_REDUCER_REDUCE(cilk_c_reducer_opmul,tn,l,r)          \
        { *(t*)l *= *(t*)r; }                                              \
    __CILKRTS_DECLARE_REDUCER_IDENTITY(cilk_c_reducer_opmul,tn)            \
        { *(t*)v = 1; }
 
//@{
/** @def CILK_C_REDUCER_OPMUL_INSTANCE 
 *  @brief Declare or define implementation functions for a reducer type.
 *
 *  In the runtime source file c_reducers.c, the macro `CILK_C_DEFINE_REDUCERS`
 *  will be defined, and this macro will generate reducer implementation
 *  functions. Everywhere else, `CILK_C_DEFINE_REDUCERS` will be undefined, and
 *  this macro will expand into external declarations for the functions.
 */
#ifdef CILK_C_DEFINE_REDUCERS
#   define CILK_C_REDUCER_OPMUL_INSTANCE(t,tn)  \
        CILK_C_REDUCER_OPMUL_DEFINITION(t,tn)
#else
#   define CILK_C_REDUCER_OPMUL_INSTANCE(t,tn)  \
        CILK_C_REDUCER_OPMUL_DECLARATION(t,tn)
#endif
//@}

/*  Declare or define an instance of the reducer type and its functions for each 
 *  numeric type.
 */
CILK_C_REDUCER_OPMUL_INSTANCE(char,                 char)
CILK_C_REDUCER_OPMUL_INSTANCE(unsigned char,        uchar)
CILK_C_REDUCER_OPMUL_INSTANCE(signed char,          schar)
CILK_C_REDUCER_OPMUL_INSTANCE(wchar_t,              wchar_t)
CILK_C_REDUCER_OPMUL_INSTANCE(short,                short)
CILK_C_REDUCER_OPMUL_INSTANCE(unsigned short,       ushort)
CILK_C_REDUCER_OPMUL_INSTANCE(int,                  int)
CILK_C_REDUCER_OPMUL_INSTANCE(unsigned int,         uint)
CILK_C_REDUCER_OPMUL_INSTANCE(unsigned int,         unsigned) /* alternate name */
CILK_C_REDUCER_OPMUL_INSTANCE(long,                 long)
CILK_C_REDUCER_OPMUL_INSTANCE(unsigned long,        ulong)
CILK_C_REDUCER_OPMUL_INSTANCE(long long,            longlong)
CILK_C_REDUCER_OPMUL_INSTANCE(unsigned long long,   ulonglong)
CILK_C_REDUCER_OPMUL_INSTANCE(float,                float)
CILK_C_REDUCER_OPMUL_INSTANCE(double,               double)
CILK_C_REDUCER_OPMUL_INSTANCE(long double,          longdouble)

//@endcond

__CILKRTS_END_EXTERN_C

//@}

//@}

#endif /*  REDUCER_OPMUL_H_INCLUDED */
