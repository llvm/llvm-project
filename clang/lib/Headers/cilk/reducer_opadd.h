/*  reducer_opadd.h                  -*- C++ -*-
 *
 *  @copyright
 *  Copyright (C) 2009-2013, Intel Corporation
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

/** @file reducer_opadd.h
 *
 *  @brief Defines classes for doing parallel addition reductions.
 *
 *  @ingroup ReducersAdd
 *
 *  @see ReducersAdd
 */

#ifndef REDUCER_OPADD_H_INCLUDED
#define REDUCER_OPADD_H_INCLUDED

#include <cilk/reducer.h>

/** @defgroup ReducersAdd Addition Reducers
 *
 *  Addition reducers allow the computation of the sum of a set of values in
 *  parallel.
 *
 *  @ingroup Reducers
 *
 *  You should be familiar with @ref pagereducers "Cilk reducers", described in
 *  file `reducers.md`, and particularly with @ref reducers_using, before trying
 *  to use the information in this file.
 *
 *  @section redopadd_usage Usage Example
 *
 *      cilk::reducer< cilk::op_add<int> > r;
 *      cilk_for (int i = 0; i != N; ++i) {
 *          *r += a[i];
 *      }
 *      return r.get_value();
 *
 *  @section redopadd_monoid The Monoid
 *
 *  @subsection redopadd_monoid_values Value Set
 *
 *  The value set of an addition reducer is the set of values of `Type`, which
 *  is expected to be a builtin numeric type (or something like it, such as
 *  `std::complex`).
 *
 *  @subsection redopadd_monoid_operator Operator
 *
 *  The operator of an addition reducer is the addition operator, defined by
 *  the “`+`” binary operator on `Type`.
 *
 *  @subsection redopadd_monoid_identity Identity
 *
 *  The identity value of the reducer is the numeric value “`0`”. This is
 *  expected to be the value of the default constructor `Type()`.
 *
 *  @section redopadd_operations Operations
 *
 *  @subsection redopadd_constructors Constructors
 *
 *      reducer()   // identity
 *      reducer(const Type& value)
 *      reducer(move_in(Type& variable))
 *
 *  @subsection redopadd_get_set Set and Get
 *
 *      r.set_value(const Type& value)
 *      const Type& = r.get_value() const
 *      r.move_in(Type& variable)
 *      r.move_out(Type& variable)
 *
 *  @subsection redopadd_initial Initial Values
 *
 *  If an addition reducer is constructed without an explicit initial value,
 *  then its initial value will be its identity value, as long as `Type`
 *  satisfies the requirements of @ref redopadd_types.
 *
 *  @subsection redopadd_view_ops View Operations
 *
 *      *r += a
 *      *r -= a
 *      ++*r
 *      --*r
 *      (*r)++
 *      (*r)--
 *      *r = *r + a
 *      *r = *r - a
 *      *r = *r ± a1 ± a2 … ± an
 *
 *  The post-increment and post-decrement operations do not return a value. (If
 *  they did, they would expose the value contained in the view, which is
 *  non-deterministic in the middle of a reduction.)
 *
 *  Note that subtraction operations are allowed on an addition reducer because
 *  subtraction is equivalent to addition with a negated operand. It is true
 *  that `(x - y) - z` is not equivalent to `x - (y - z)`, but
 *  `(x + (-y)) + (-z)` _is_ equivalent to `x + ((-y) + (-z))`.
 *
 *  @section redopadd_floating_point Issues with Floating-Point Types
 *
 *  Because of precision and round-off issues, floating-point addition is not
 *  really associative. For example, `(1e30 + -1e30) + 1 == 1`, but 
 *  `1e30 + (-1e30 + 1) == 0`.
 *
 *  In many cases, this won’t matter, but computations which have been
 *  carefully ordered to control round-off errors may not deal well with
 *  being reassociated. In general, you should be sure to understand the
 *  floating-point behavior of your program before doing any transformation 
 *  that will reassociate its computations. 
 *
 *  @section redopadd_types Type and Operator Requirements
 *
 *  `Type` must be `Copy Constructible`, `Default Constructible`, and
 *  `Assignable`.
 *
 *  The operator “`+=`” must be defined on `Type`, with `x += a` having the
 *  same meaning as `x = x + a`. In addition, if the code uses the “`-=`”,
 *  pre-increment, post-increment, pre-decrement, or post-decrement operators,
 *  then the corresponding operators must be defined on `Type`.
 *
 *  The expression `Type()` must be a valid expression which yields the
 *  identity value (the value of `Type` whose numeric value is zero).
 *
 *  @section redopadd_in_c Addition Reducers in C
 *
 *  The @ref CILK_C_REDUCER_OPADD and @ref CILK_C_REDUCER_OPADD_TYPE macros can
 *  be used to do addition reductions in C. For example:
 *
 *      CILK_C_REDUCER_OPADD(r, double, 0);
 *      CILK_C_REGISTER_REDUCER(r);
 *      cilk_for(int i = 0; i != n; ++i) {
 *          REDUCER_VIEW(r) += a[i];
 *      }
 *      CILK_C_UNREGISTER_REDUCER(r);
 *      printf("The sum of the elements of a is %f\n", REDUCER_VIEW(r));
 *
 *  See @ref reducers_c_predefined.
 */

#ifdef __cplusplus

namespace cilk {

/** The addition reducer view class.
 *
 *  This is the view class for reducers created with 
 *  `cilk::reducer< cilk::op_add<Type> >`. It holds the accumulator variable 
 *  for the reduction, and allows only addition and subtraction operations to 
 *  be performed on it.
 *
 *  @note   The reducer “dereference” operation (`reducer::operator *()`) 
 *          yields a reference to the view. Thus, for example, the view class’s
 *          `+=` operation would be used in an expression like `*r += a`, where
 *          `r` is an op_add reducer variable.
 *
 *  @tparam Type    The type of the contained accumulator variable. This will 
 *                  be the value type of a monoid_with_view that is 
 *                  instantiated with this view.
 *
 *  @see ReducersAdd
 *  @see op_add
 *
 *  @ingroup ReducersAdd
 */
template <typename Type>
class op_add_view : public scalar_view<Type>
{
    typedef scalar_view<Type> base;
    
public:
    /** Class to represent the right-hand side of 
     *  `*reducer = *reducer ± value`.
     *
     *  The only assignment operator for the op_add_view class takes an
     *  rhs_proxy as its operand. This results in the syntactic restriction
     *  that the only expressions that can be assigned to an op_add_view are
     *  ones which generate an rhs_proxy — that is, expressions of the form
     *  `op_add_view ± value ... ± value`.
     *
     *  @warning
     *  The lhs and rhs views in such an assignment must be the same; 
     *  otherwise, the behavior will be undefined. (I.e., `v1 = v1 + x` is
     *  legal; `v1 = v2 + x` is illegal.) This condition will be checked with a
     *  runtime assertion when compiled in debug mode.
     *
     *  @see op_add_view
     */
    class rhs_proxy {
        friend class op_add_view;

        const op_add_view* m_view;
        Type               m_value;

        // Constructor is invoked only from op_add_view::operator+() and 
        // op_add_view::operator-().
        //
        rhs_proxy(const op_add_view* view, const Type& value) :
            m_view(view), m_value(value) {}

        rhs_proxy& operator=(const rhs_proxy&); // Disable assignment operator
        rhs_proxy();                            // Disable default constructor

    public:
        //@{
        /** Add or subtract an additional rhs value. If `v` is an op_add_view
         *  and `a1` is a value, then the expression `v + a1` invokes the view’s
         *  `operator+()` to create an rhs_proxy for `(v, a1)`; then 
         *  `v + a1 + a2` invokes the rhs_proxy’s `operator+()` to create a new
         *  rhs_proxy for `(v, a1+a2)`. This allows the right-hand side of an
         *  assignment to be not just `view ± value`, but 
         *  `view ± value ± value ... ± value`. The effect is that
         *
         *      v = v ± a1 ± a2 ... ± an;
         *
         *  is evaluated as
         *
         *      v = v ± (±a1 ± a2 ... ± an);
         */
        rhs_proxy& operator+(const Type& x) { m_value += x; return *this; }
        rhs_proxy& operator-(const Type& x) { m_value -= x; return *this; }
        //@}
    };

    
    /** Default/identity constructor. This constructor initializes the 
     *  contained value to `Type()`, which is expected to be the identity value
     *  for addition on `Type`.
     */
    op_add_view() : base() {}

    /** Construct with a specified initial value.
     */
    explicit op_add_view(const Type& v) : base(v) {}
    
    /** Reduction operation.
     *
     *  This function is invoked by the @ref op_add monoid to combine the views
     *  of two strands when the right strand merges with the left one. It adds
     *  the value contained in the right-strand view to the value contained in
     *  the left-strand view, and leaves the value in the right-strand view
     *  undefined.
     *
     *  @param  right   A pointer to the right-strand view. (`this` points to
     *                  the left-strand view.)
     *
     *  @note   Used only by the @ref op_add monoid to implement the monoid
     *          reduce operation.
     */
    void reduce(op_add_view* right) { this->m_value += right->m_value; }

    /** @name Accumulator variable updates.
     *
     *  These functions support the various syntaxes for incrementing or
     *  decrementing the accumulator variable contained in the view.
     */
    //@{

    /** Increment the accumulator variable by @a x.
     */
    op_add_view& operator+=(const Type& x) { this->m_value += x; return *this; }

    /** Decrement the accumulator variable by @a x.
     */
    op_add_view& operator-=(const Type& x) { this->m_value -= x; return *this; }

    /** Pre-increment.
     */
    op_add_view& operator++() { ++this->m_value; return *this; }

    /** Post-increment.
     *
     *  @note   Conventionally, post-increment operators return the old value
     *          of the incremented variable. However, reducer views do not
     *          expose their contained values, so `view++` does not have a
     *          return value.
     */
    void operator++(int) { this->m_value++; }

    /** Pre-decrement.
     */
    op_add_view& operator--() { --this->m_value; return *this; }

    /** Post-decrement.
     *
     *  @note   Conventionally, post-decrement operators return the old value
     *          of the decremented variable. However, reducer views do not
     *          expose their contained values, so `view--` does not have a
     *          return value.
     */
    void operator--(int) { this->m_value--; }

    /** Create an object representing `*this + x`.
     *
     *  @see rhs_proxy
     */
    rhs_proxy operator+(const Type& x) const { return rhs_proxy(this, x); }

    /** Create an object representing `*this - x`.
     *
     *  @see rhs_proxy
     */
    rhs_proxy operator-(const Type& x) const { return rhs_proxy(this, -x); }

    /** Assign the result of a `view ± value` expression to the view. Note that
     *  this is the only assignment operator for this class.
     *
     *  @see rhs_proxy
     */
    op_add_view& operator=(const rhs_proxy& rhs) {
        __CILKRTS_ASSERT(this == rhs.m_view);
        this->m_value += rhs.m_value;
        return *this;
    }
    
    //@}
};


/** Monoid class for addition reductions. Instantiate the cilk::reducer 
 *  template class with an op_add monoid to create an addition reducer class.
 *  For example, to compute
 *  the sum of a set of `int` values:
 *
 *      cilk::reducer< cilk::op_add<int> > r;
 *
 *  @tparam Type    The reducer value type.
 *  @tparam Align   If `false` (the default), reducers instantiated on this
 *                  monoid will be naturally aligned (the Cilk library 1.0
 *                  behavior). If `true`, reducers instantiated on this monoid
 *                  will be cache-aligned for binary compatibility with 
 *                  reducers in Cilk library version 0.9.
 *
 *  @see ReducersAdd
 *  @see op_add_view
 *
 *  @ingroup ReducersAdd
 */
template <typename Type, bool Align = false>
struct op_add : public monoid_with_view<op_add_view<Type>, Align> {};

/** **Deprecated** addition reducer wrapper class.
 *
 *  reducer_opadd is the same as @ref reducer<@ref op_add>, except that
 *  reducer_opadd is a proxy for the contained view, so that accumulator
 *  variable update operations can be applied directly to the reducer. For
 *  example, a value is added to a `reducer<%op_add>` with `*r += a`, but a
 *  value can be added to a `%reducer_opadd` with `r += a`.
 *
 *  @deprecated Users are strongly encouraged to use `reducer<monoid>`
 *              reducers rather than the old wrappers like reducer_opadd. 
 *              The `reducer<monoid>` reducers show the reducer/monoid/view
 *              architecture more clearly, are more consistent in their
 *              implementation, and present a simpler model for new
 *              user-implemented reducers.
 *
 *  @note   Implicit conversions are provided between `%reducer_opadd` 
 *          and `reducer<%op_add>`. This allows incremental code
 *          conversion: old code that used `%reducer_opadd` can pass a
 *          `%reducer_opadd` to a converted function that now expects a
 *          pointer or reference to a `reducer<%op_add>`, and vice
 *          versa.
 *
 *  @tparam Type    The value type of the reducer.
 *
 *  @see op_add
 *  @see reducer
 *  @see ReducersAdd
 *
 *  @ingroup ReducersAdd
 */
template <typename Type>
class reducer_opadd : public reducer< op_add<Type, true> >
{
    typedef reducer< op_add<Type, true> > base;
    using base::view;

  public:
    /// The view type for the reducer.
    typedef typename base::view_type        view_type;
    
    /// The view’s rhs proxy type.
    typedef typename view_type::rhs_proxy   rhs_proxy;

    /// The view type for the reducer.
    typedef view_type                       View;

    /// The monoid type for the reducer.
    typedef typename base::monoid_type      Monoid;

    /** @name Constructors
     */
    //@{
    
    /** Default (identity) constructor.
     *
     * Constructs the wrapper with the default initial value of `Type()`.
     */
    reducer_opadd() {}

    /** Value constructor.
     *
     *  Constructs the wrapper with a specified initial value.
     */
    explicit reducer_opadd(const Type& initial_value) : base(initial_value) {}
    
    //@}

    /** @name Forwarded functions
     *  @details Functions that update the contained accumulator variable are
     *  simply forwarded to the contained @ref op_add_view. */
    //@{
    
    /// @copydoc op_add_view::operator+=(const Type&)
    reducer_opadd& operator+=(const Type& x)    { view() += x; return *this; }
    
    /// @copydoc op_add_view::operator-=(const Type&)
    reducer_opadd& operator-=(const Type& x)    { view() -= x; return *this; }
    
    /// @copydoc op_add_view::operator++()
    reducer_opadd& operator++()                 { ++view(); return *this; }
    
    /// @copydoc op_add_view::operator++(int)
    void operator++(int)                        { view()++; }
    
    /// @copydoc op_add_view::operator-\-()
    reducer_opadd& operator--()                 { --view(); return *this; }
    
    /// @copydoc op_add_view::operator-\-(int)
    void operator--(int)                        { view()--; }

    // The legacy definitions of reducer_opadd::operator+() and
    // reducer_opadd::operator-() have different behavior and a different
    // return type than this definition. The legacy version is defined as a
    // member function, so this new version is defined as a free function to
    // give it a different signature, so that they won’t end up sharing a
    // single object file entry.

    /// @copydoc op_add_view::operator+(const Type&) const
    friend rhs_proxy operator+(const reducer_opadd& r, const Type& x)
    { 
        return r.view() + x; 
    }
    /// @copydoc op_add_view::operator-(const Type&) const
    friend rhs_proxy operator-(const reducer_opadd& r, const Type& x)
    { 
        return r.view() - x; 
    }
    /// @copydoc op_add_view::operator=(const rhs_proxy&)
    reducer_opadd& operator=(const rhs_proxy& temp) 
    {
        view() = temp;
        return *this; 
    }
    //@}

    /** @name Dereference
     *  @details Dereferencing a wrapper is a no-op. It simply returns the
     *  wrapper. Combined with the rule that the wrapper forwards view
     *  operations to its contained view, this means that view operations can
     *  be written the same way on reducers and wrappers, which is convenient
     *  for incrementally converting old code using wrappers to use reducers
     *  instead. That is:
     *
     *      reducer< op_add<int> > r;
     *      *r += a;    // *r returns the view
     *                  // operator += is a view member function
     *
     *      reducer_opadd<int> w;
     *      *w += a;    // *w returns the wrapper
     *                  // operator += is a wrapper member function that
     *                  // calls the corresponding view function
     */
    //@{
    reducer_opadd&       operator*()       { return *this; }
    reducer_opadd const& operator*() const { return *this; }

    reducer_opadd*       operator->()       { return this; }
    reducer_opadd const* operator->() const { return this; }
    //@}
    
    /** @name Upcast
     *  @details In Cilk library 0.9, reducers were always cache-aligned. In
     *  library  1.0, reducer cache alignment is optional. By default, reducers
     *  are unaligned (i.e., just naturally aligned), but legacy wrappers
     *  inherit from cache-aligned reducers for binary compatibility.
     *
     *  This means that a wrapper will automatically be upcast to its aligned
     *  reducer base class. The following conversion operators provide
     *  pseudo-upcasts to the corresponding unaligned reducer class.
     */
    //@{
    operator reducer< op_add<Type, false> >& ()
    {
        return *reinterpret_cast< reducer< op_add<Type, false> >* >(this);
    }
    operator const reducer< op_add<Type, false> >& () const
    {
        return *reinterpret_cast< const reducer< op_add<Type, false> >* >(this);
    }
    //@}
};

/// @cond internal
/** Metafunction specialization for reducer conversion.
 *
 *  This specialization of the @ref legacy_reducer_downcast template class 
 *  defined in reducer.h causes the `reducer< op_add<Type> >` class to have an 
 *  `operator reducer_opadd<Type>& ()` conversion operator that statically 
 *  downcasts the `reducer<op_add>` to the corresponding `reducer_opadd` type.
 *  (The reverse conversion, from `reducer_opadd` to `reducer<op_add>`, is just
 *  an upcast, which is provided for free by the language.)
 *
 *  @ingroup ReducersAdd
 */
template <typename Type, bool Align>
struct legacy_reducer_downcast<reducer<op_add<Type, Align> > >
{
    typedef reducer_opadd<Type> type;
};
/// @endcond

} // namespace cilk

#endif // __cplusplus


/** @ingroup ReducersAdd
 */
//@{

/** @name C Language Reducer Macros
 *
 *  These macros are used to declare and work with numeric op_add reducers in 
 *  C code.
 *
 *  @see @ref page_reducers_in_c
 */
 //@{
 
__CILKRTS_BEGIN_EXTERN_C

/** Opadd reducer type name.
 *
 *  This macro expands into the identifier which is the name of the op_add
 *  reducer type for a specified numeric type.
 *
 *  @param  tn  The @ref reducers_c_type_names "numeric type name" specifying
 *              the type of the reducer.
 *
 *  @see @ref reducers_c_predefined
 *  @see ReducersAdd
 */
#define CILK_C_REDUCER_OPADD_TYPE(tn)                                         \
    __CILKRTS_MKIDENT(cilk_c_reducer_opadd_,tn)

/** Declare an op_add reducer object.
 *
 *  This macro expands into a declaration of an op_add reducer object for a
 *  specified numeric type. For example:
 *
 *      CILK_C_REDUCER_OPADD(my_reducer, double, 0.0);
 *
 *  @param  obj The variable name to be used for the declared reducer object.
 *  @param  tn  The @ref reducers_c_type_names "numeric type name" specifying
 *              the type of the reducer.
 *  @param  v   The initial value for the reducer. (A value which can be
 *              assigned to the numeric type represented by @a tn.)
 *
 *  @see @ref reducers_c_predefined
 *  @see ReducersAdd
 */
#define CILK_C_REDUCER_OPADD(obj,tn,v)                                        \
    CILK_C_REDUCER_OPADD_TYPE(tn) obj =                                       \
        CILK_C_INIT_REDUCER(_Typeof(obj.value),                               \
                        __CILKRTS_MKIDENT(cilk_c_reducer_opadd_reduce_,tn),   \
                        __CILKRTS_MKIDENT(cilk_c_reducer_opadd_identity_,tn), \
                        __cilkrts_hyperobject_noop_destroy, v)

/// @cond internal

/** Declare the op_add reducer functions for a numeric type.
 *
 *  This macro expands into external function declarations for functions which
 *  implement the reducer functionality for the op_add reducer type for a
 *  specified numeric type.
 *
 *  @param  t   The value type of the reducer.
 *  @param  tn  The value “type name” identifier, used to construct the reducer
 *              type name, function names, etc.
 */
#define CILK_C_REDUCER_OPADD_DECLARATION(t,tn)                             \
    typedef CILK_C_DECLARE_REDUCER(t) CILK_C_REDUCER_OPADD_TYPE(tn);       \
    __CILKRTS_DECLARE_REDUCER_REDUCE(cilk_c_reducer_opadd,tn,l,r);         \
    __CILKRTS_DECLARE_REDUCER_IDENTITY(cilk_c_reducer_opadd,tn);
 
/** Define the op_add reducer functions for a numeric type.
 *
 *  This macro expands into function definitions for functions which implement
 *  the reducer functionality for the op_add reducer type for a specified
 *  numeric type.
 *
 *  @param  t   The value type of the reducer.
 *  @param  tn  The value “type name” identifier, used to construct the reducer
 *              type name, function names, etc.
 */
#define CILK_C_REDUCER_OPADD_DEFINITION(t,tn)                              \
    typedef CILK_C_DECLARE_REDUCER(t) CILK_C_REDUCER_OPADD_TYPE(tn);       \
    __CILKRTS_DECLARE_REDUCER_REDUCE(cilk_c_reducer_opadd,tn,l,r)          \
        { *(t*)l += *(t*)r; }                                              \
    __CILKRTS_DECLARE_REDUCER_IDENTITY(cilk_c_reducer_opadd,tn)            \
        { *(t*)v = 0; }
 
//@{
/** @def CILK_C_REDUCER_OPADD_INSTANCE 
 *  @brief Declare or define implementation functions for a reducer type.
 *
 *  In the runtime source file c_reducers.c, the macro `CILK_C_DEFINE_REDUCERS`
 *  will be defined, and this macro will generate reducer implementation 
 *  functions. Everywhere else, `CILK_C_DEFINE_REDUCERS` will be undefined,
 *  and this macro will expand into external declarations for the functions.
 */
#ifdef CILK_C_DEFINE_REDUCERS
#   define CILK_C_REDUCER_OPADD_INSTANCE(t,tn)  \
        CILK_C_REDUCER_OPADD_DEFINITION(t,tn)
#else
#   define CILK_C_REDUCER_OPADD_INSTANCE(t,tn)  \
        CILK_C_REDUCER_OPADD_DECLARATION(t,tn)
#endif
//@}

/*  Declare or define an instance of the reducer type and its functions for each 
 *  numeric type.
 */
CILK_C_REDUCER_OPADD_INSTANCE(char,                 char)
CILK_C_REDUCER_OPADD_INSTANCE(unsigned char,        uchar)
CILK_C_REDUCER_OPADD_INSTANCE(signed char,          schar)
CILK_C_REDUCER_OPADD_INSTANCE(wchar_t,              wchar_t)
CILK_C_REDUCER_OPADD_INSTANCE(short,                short)
CILK_C_REDUCER_OPADD_INSTANCE(unsigned short,       ushort)
CILK_C_REDUCER_OPADD_INSTANCE(int,                  int)
CILK_C_REDUCER_OPADD_INSTANCE(unsigned int,         uint)
CILK_C_REDUCER_OPADD_INSTANCE(unsigned int,         unsigned) /* alternate name */
CILK_C_REDUCER_OPADD_INSTANCE(long,                 long)
CILK_C_REDUCER_OPADD_INSTANCE(unsigned long,        ulong)
CILK_C_REDUCER_OPADD_INSTANCE(long long,            longlong)
CILK_C_REDUCER_OPADD_INSTANCE(unsigned long long,   ulonglong)
CILK_C_REDUCER_OPADD_INSTANCE(float,                float)
CILK_C_REDUCER_OPADD_INSTANCE(double,               double)
CILK_C_REDUCER_OPADD_INSTANCE(long double,          longdouble)

//@endcond

__CILKRTS_END_EXTERN_C

//@}

//@}

#endif /*  REDUCER_OPADD_H_INCLUDED */
