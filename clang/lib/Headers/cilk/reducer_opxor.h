/*  reducer_opxor.h                  -*- C++ -*-
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

/** @file reducer_opxor.h
 *
 *  @brief Defines classes for doing parallel bitwise or reductions.
 *
 *  @ingroup ReducersXor
 *
 *  @see ReducersXor
 */

#ifndef REDUCER_OPXOR_H_INCLUDED
#define REDUCER_OPXOR_H_INCLUDED

#include <cilk/reducer.h>

/** @defgroup ReducersXor Bitwise Xor Reducers
 *
 *  Bitwise and reducers allow the computation of the bitwise and of a set of
 *  values in parallel.
 *
 *  @ingroup Reducers
 *
 *  You should be familiar with @ref pagereducers "Cilk reducers", described in
 *  file `reducers.md`, and particularly with @ref reducers_using, before trying
 *  to use the information in this file.
 *
 *  @section redopxor_usage Usage Example
 *
 *      cilk::reducer< cilk::op_xor<unsigned> > r;
 *      cilk_for (int i = 0; i != N; ++i) {
 *          *r ^= a[i];
 *      }
 *      unsigned result;
 *      r.move_out(result);
 *
 *  @section redopxor_monoid The Monoid
 *
 *  @subsection redopxor_monoid_values Value Set
 *
 *  The value set of a bitwise xor reducer is the set of values of `Type`, which
 *  is expected to be a builtin integer type which has a representation as a
 *  sequence of bits (or something like it, such as `bool` or `std::bitset`).
 *
 *  @subsection redopxor_monoid_operator Operator
 *
 *  The operator of a bitwise xor reducer is the bitwise xor operator, defined
 *  by the “`^`” binary operator on `Type`.
 *
 *  @subsection redopxor_monoid_identity Identity
 *
 *  The identity value of the reducer is the value whose representation 
 *  contains all 0-bits. This is expected to be the value of the default
 *  constructor `Type()`.
 *
 *  @section redopxor_operations Operations
 *
 *  @subsection redopxor_constructors Constructors
 *
 *      reducer()   // identity
 *      reducer(const Type& value)
 *      reducer(move_in(Type& variable))
 *
 *  @subsection redopxor_get_set Set and Get
 *
 *      r.set_value(const Type& value)
 *      const Type& = r.get_value() const
 *      r.move_in(Type& variable)
 *      r.move_out(Type& variable)
 *
 *  @subsection redopxor_initial Initial Values
 *
 *  If a bitwise xor reducer is constructed without an explicit initial value, 
 *  then its initial value will be its identity value, as long as `Type` 
 *  satisfies the requirements of @ref redopxor_types.
 *
 *  @subsection redopxor_view_ops View Operations
 *
 *      *r ^= a
 *      *r = *r ^ a
 *      *r = *r ^ a1 ^ a2 … ^ an
 *
 *  @section redopxor_types Type and Operator Requirements
 *
 *  `Type` must be `Copy Constructible`, `Default Constructible`, and
 *  `Assignable`.
 *
 *  The operator “`^=`” must be defined on `Type`, with `x ^= a` having the 
 *  same meaning as `x = x ^ a`.
 *
 *  The expression `Type()` must be a valid expression which yields the
 *  identity value (the value of `Type` whose representation consists of all
 *  0-bits).
 *
 *  @section redopxor_in_c Bitwise Xor Reducers in C
 *
 *  The @ref CILK_C_REDUCER_OPXOR and @ref CILK_C_REDUCER_OPXOR_TYPE macros can
 *  be used to do bitwise xor reductions in C. For example:
 *
 *      CILK_C_REDUCER_OPXOR(r, uint, 0);
 *      CILK_C_REGISTER_REDUCER(r);
 *      cilk_for(int i = 0; i != n; ++i) {
 *          REDUCER_VIEW(r) ^= a[i];
 *      }
 *      CILK_C_UNREGISTER_REDUCER(r);
 *      printf("The bitwise XOR of the elements of a is %x\n", REDUCER_VIEW(r));
 *
 *  See @ref reducers_c_predefined.
 */

#ifdef __cplusplus

namespace cilk {

/** The bitwise xor reducer view class.
 *
 *  This is the view class for reducers created with 
 *  `cilk::reducer< cilk::op_xor<Type> >`. It holds the accumulator variable 
 *  for the reduction, and allows only `xor` operations to be performed on it.
 *
 *  @note   The reducer “dereference” operation (`reducer::operator *()`) 
 *          yields a reference to the view. Thus, for example, the view class’s
 *          `^=` operation would be used in an expression like `*r ^= a`, where
 *          `r` is an opmod reducer variable.
 *
 *  @tparam Type    The type of the contained accumulator variable. This will
 *                  be the value type of a monoid_with_view that is
 *                  instantiated with this view.
 *
 *  @see ReducersXor
 *  @see op_xor
 *
 *  @ingroup ReducersXor
 */
template <typename Type>
class op_xor_view : public scalar_view<Type>
{
    typedef scalar_view<Type> base;
    
public:
    /** Class to represent the right-hand side of `*reducer = *reducer ^ value`.
     *
     *  The only assignment operator for the op_xor_view class takes an 
     *  rhs_proxy as its operand. This results in the syntactic restriction
     *  that the only expressions that can be assigned to an op_xor_view are
     *  ones which generate an rhs_proxy — that is, expressions of the form
     *  `op_xor_view ^ value ... ^ value`.
     *
     *  @warning
     *  The lhs and rhs views in such an assignment must be the same; 
     *  otherwise, the behavior will be undefined. (I.e., `v1 = v1 ^ x` is
     *  legal; `v1 = v2 ^ x` is illegal.) This condition will be checked with
     *  a runtime assertion when compiled in debug mode.
     *
     *  @see op_xor_view
     */
    class rhs_proxy {
        friend class op_xor_view;

        const op_xor_view* m_view;
        Type              m_value;

        // Constructor is invoked only from op_xor_view::operator^().
        //
        rhs_proxy(const op_xor_view* view, const Type& value) : m_view(view), m_value(value) {}

        rhs_proxy& operator=(const rhs_proxy&); // Disable assignment operator
        rhs_proxy();                            // Disable default constructor

    public:
        /** Bitwise xor with an additional rhs value. If `v` is an op_xor_view
         *  and `a1` is a value, then the expression `v ^ a1` invokes the 
         *  view’s `operator^()` to create an rhs_proxy for `(v, a1)`; then 
         *  `v ^ a1 ^ a2` invokes the rhs_proxy’s `operator^()` to create a new
         *  rhs_proxy for `(v, a1^a2)`. This allows the right-hand side of an
         *  assignment to be not just `view ^ value`, but 
         (  `view ^ value ^ value ... ^ value`. The effect is that
         *
         *      v = v ^ a1 ^ a2 ... ^ an;
         *
         *  is evaluated as
         *
         *      v = v ^ (a1 ^ a2 ... ^ an);
         */
        rhs_proxy& operator^(const Type& x) { m_value ^= x; return *this; }
    };


    /** Default/identity constructor. This constructor initializes the
     *  contained value to `Type()`.
     */
    op_xor_view() : base() {}

    /** Construct with a specified initial value.
     */
    explicit op_xor_view(const Type& v) : base(v) {}
    
    /** Reduction operation.
     *
     *  This function is invoked by the @ref op_xor monoid to combine the views
     *  of two strands when the right strand merges with the left one. It
     *  “xors” the value contained in the left-strand view by the value
     *  contained in the right-strand view, and leaves the value in the
     *  right-strand view undefined.
     *
     *  @param  right   A pointer to the right-strand view. (`this` points to
     *                  the left-strand view.)
     *
     *  @note   Used only by the @ref op_xor monoid to implement the monoid
     *          reduce operation.
     */
    void reduce(op_xor_view* right) { this->m_value ^= right->m_value; }
    
    /** @name Accumulator variable updates.
     *
     *  These functions support the various syntaxes for “xoring” the
     *  accumulator variable contained in the view with some value.
     */
    //@{

    /** Xor the accumulator variable with @a x.
     */
    op_xor_view& operator^=(const Type& x) { this->m_value ^= x; return *this; }

    /** Create an object representing `*this ^ x`.
     *
     *  @see rhs_proxy
     */
    rhs_proxy operator^(const Type& x) const { return rhs_proxy(this, x); }

    /** Assign the result of a `view ^ value` expression to the view. Note that
     *  this is the only assignment operator for this class.
     *
     *  @see rhs_proxy
     */
    op_xor_view& operator=(const rhs_proxy& rhs) {
        __CILKRTS_ASSERT(this == rhs.m_view);
        this->m_value ^= rhs.m_value;
        return *this;
    }
    
    //@}
};

/** Monoid class for bitwise xor reductions. Instantiate the cilk::reducer 
 *  template class with an op_xor monoid to create a bitwise xor reducer
 *  class. For example, to compute the bitwise xor of a set of `unsigned long`
 *  values:
 *
 *      cilk::reducer< cilk::op_xor<unsigned long> > r;
 *
 *  @tparam Type    The reducer value type.
 *  @tparam Align   If `false` (the default), reducers instantiated on this
 *                  monoid will be naturally aligned (the Cilk library 1.0
 *                  behavior). If `true`, reducers instantiated on this monoid
 *                  will be cache-aligned for binary compatibility with 
 *                  reducers in Cilk library version 0.9.
 *
 *  @see ReducersXor
 *  @see op_xor_view
 *
 *  @ingroup ReducersXor
 */
template <typename Type, bool Align = false>
struct op_xor : public monoid_with_view<op_xor_view<Type>, Align> {};

/** Deprecated bitwise xor reducer class.
 *
 *  reducer_opxor is the same as @ref reducer<@ref op_xor>, except that
 *  reducer_opxor is a proxy for the contained view, so that accumulator
 *  variable update operations can be applied directly to the reducer. For
 *  example, a value is xored with  a `reducer<%op_xor>` with `*r ^= a`, but a
 *  value can be xored with a `%reducer_opxor` with `r ^= a`.
 *
 *  @deprecated Users are strongly encouraged to use `reducer<monoid>`
 *              reducers rather than the old wrappers like reducer_opand. 
 *              The `reducer<monoid>` reducers show the reducer/monoid/view
 *              architecture more clearly, are more consistent in their
 *              implementation, and present a simpler model for new
 *              user-implemented reducers.
 *
 *  @note   Implicit conversions are provided between `%reducer_opxor` 
 *          and `reducer<%op_xor>`. This allows incremental code
 *          conversion: old code that used `%reducer_opxor` can pass a
 *          `%reducer_opxor` to a converted function that now expects a
 *          pointer or reference to a `reducer<%op_xor>`, and vice
 *          versa.
 *
 *  @tparam Type    The value type of the reducer.
 *
 *  @see op_xor
 *  @see reducer
 *  @see ReducersXor
 *
 *  @ingroup ReducersXor
 */
template <typename Type>
class reducer_opxor : public reducer< op_xor<Type, true> >
{
    typedef reducer< op_xor<Type, true> > base;
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
    reducer_opxor() {}

    /** Value constructor.
     *
     *  Constructs the wrapper with a specified initial value.
     */
    explicit reducer_opxor(const Type& initial_value) : base(initial_value) {}
    
    //@}

    /** @name Forwarded functions
     *  @details Functions that update the contained accumulator variable are
     *  simply forwarded to the contained @ref op_and_view. */
    //@{

    /// @copydoc op_xor_view::operator^=(const Type&)
    reducer_opxor& operator^=(const Type& x)
    {
        view() ^= x; return *this; 
    }
    
    // The legacy definition of reducer_opxor::operator^() has different
    // behavior and a different return type than this definition. The legacy
    // version is defined as a member function, so this new version is defined
    // as a free function to give it a different signature, so that they won’t 
    // end up sharing a single object file entry.

    /// @copydoc op_xor_view::operator^(const Type&) const
    friend rhs_proxy operator^(const reducer_opxor& r, const Type& x)
    { 
        return r.view() ^ x; 
    }

    /// @copydoc op_and_view::operator=(const rhs_proxy&)
    reducer_opxor& operator=(const rhs_proxy& temp)
    {
        view() = temp; return *this; 
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
     *      reducer< op_and<int> > r;
     *      *r &= a;    // *r returns the view
     *                  // operator &= is a view member function
     *
     *      reducer_opand<int> w;
     *      *w &= a;    // *w returns the wrapper
     *                  // operator &= is a wrapper member function that
     *                  // calls the corresponding view function
     */
    //@{
    reducer_opxor&       operator*()       { return *this; }
    reducer_opxor const& operator*() const { return *this; }

    reducer_opxor*       operator->()       { return this; }
    reducer_opxor const* operator->() const { return this; }
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
    operator reducer< op_xor<Type, false> >& ()
    {
        return *reinterpret_cast< reducer< op_xor<Type, false> >* >(this);
    }
    operator const reducer< op_xor<Type, false> >& () const
    {
        return *reinterpret_cast< const reducer< op_xor<Type, false> >* >(this);
    }
    //@}
    
};

/// @cond internal
/** Metafunction specialization for reducer conversion.
 *
 *  This specialization of the @ref legacy_reducer_downcast template class 
 *  defined in reducer.h causes the `reducer< op_xor<Type> >` class to have an 
 *  `operator reducer_opxor<Type>& ()` conversion operator that statically
 *  downcasts the `reducer<op_xor>` to the corresponding `reducer_opxor` type.
 *  (The reverse conversion, from `reducer_opxor` to `reducer<op_xor>`, is just
 *  an upcast, which is provided for free by the language.)
 *
 *  @ingroup ReducersXor
 */
template <typename Type, bool Align>
struct legacy_reducer_downcast<reducer<op_xor<Type, Align> > >
{
    typedef reducer_opxor<Type> type;
};
/// @endcond

} // namespace cilk

#endif /* __cplusplus */


/** @ingroup ReducersXor
 */
//@{

/** @name C language reducer macros
 *
 *  These macros are used to declare and work with op_xor reducers in C code.
 *
 *  @see @ref page_reducers_in_c
 */
 //@{
 
__CILKRTS_BEGIN_EXTERN_C

/** Opxor reducer type name.
 *
 *  This macro expands into the identifier which is the name of the op_xor
 *  reducer type for a specified numeric type.
 *
 *  @param  tn  The @ref reducers_c_type_names "numeric type name" specifying
 *              the type of the reducer.
 *
 *  @see @ref reducers_c_predefined
 *  @see ReducersXor
 */
#define CILK_C_REDUCER_OPXOR_TYPE(tn)                                         \
    __CILKRTS_MKIDENT(cilk_c_reducer_opxor_,tn)

/** Declare an op_xor reducer object.
 *
 *  This macro expands into a declaration of an op_xor reducer object for a
 *  specified numeric type. For example:
 *
 *      CILK_C_REDUCER_OPXOR(my_reducer, ulong, 0);
 *
 *  @param  obj The variable name to be used for the declared reducer object.
 *  @param  tn  The @ref reducers_c_type_names "numeric type name" specifying
 *              the type of the reducer.
 *  @param  v   The initial value for the reducer. (A value which can be
 *              assigned to the numeric type represented by @a tn.)
 *
 *  @see @ref reducers_c_predefined
 *  @see ReducersXor
 */
#define CILK_C_REDUCER_OPXOR(obj,tn,v)                                        \
    CILK_C_REDUCER_OPXOR_TYPE(tn) obj =                                       \
        CILK_C_INIT_REDUCER(_Typeof(obj.value),                               \
                        __CILKRTS_MKIDENT(cilk_c_reducer_opxor_reduce_,tn),   \
                        __CILKRTS_MKIDENT(cilk_c_reducer_opxor_identity_,tn), \
                        __cilkrts_hyperobject_noop_destroy, v)

/// @cond internal

/** Declare the op_xor reducer functions for a numeric type.
 *
 *  This macro expands into external function declarations for functions which
 *  implement the reducer functionality for the op_xor reducer type for a
 *  specified numeric type.
 *
 *  @param  t   The value type of the reducer.
 *  @param  tn  The value “type name” identifier, used to construct the reducer
 *              type name, function names, etc.
 */
#define CILK_C_REDUCER_OPXOR_DECLARATION(t,tn)                             \
    typedef CILK_C_DECLARE_REDUCER(t) CILK_C_REDUCER_OPXOR_TYPE(tn);       \
    __CILKRTS_DECLARE_REDUCER_REDUCE(cilk_c_reducer_opxor,tn,l,r);         \
    __CILKRTS_DECLARE_REDUCER_IDENTITY(cilk_c_reducer_opxor,tn);
 
/** Define the op_xor reducer functions for a numeric type.
 *
 *  This macro expands into function definitions for functions which implement
 *  the reducer functionality for the op_xor reducer type for a specified 
 *  numeric type.
 *
 *  @param  t   The value type of the reducer.
 *  @param  tn  The value “type name” identifier, used to construct the reducer
 *              type name, function names, etc.
 */
#define CILK_C_REDUCER_OPXOR_DEFINITION(t,tn)                              \
    typedef CILK_C_DECLARE_REDUCER(t) CILK_C_REDUCER_OPXOR_TYPE(tn);       \
    __CILKRTS_DECLARE_REDUCER_REDUCE(cilk_c_reducer_opxor,tn,l,r)          \
        { *(t*)l ^= *(t*)r; }                                              \
    __CILKRTS_DECLARE_REDUCER_IDENTITY(cilk_c_reducer_opxor,tn)            \
        { *(t*)v = 0; }
 
//@{
/** @def CILK_C_REDUCER_OPXOR_INSTANCE 
 *  @brief Declare or define implementation functions for a reducer type.
 *
 *  In the runtime source file c_reducers.c, the macro `CILK_C_DEFINE_REDUCERS`
 *  will be defined, and this macro will generate reducer implementation
 *  functions. Everywhere else, `CILK_C_DEFINE_REDUCERS` will be undefined, and
 *  this macro will expand into external declarations for the functions.
 */
#ifdef CILK_C_DEFINE_REDUCERS
#   define CILK_C_REDUCER_OPXOR_INSTANCE(t,tn)  \
        CILK_C_REDUCER_OPXOR_DEFINITION(t,tn)
#else
#   define CILK_C_REDUCER_OPXOR_INSTANCE(t,tn)  \
        CILK_C_REDUCER_OPXOR_DECLARATION(t,tn)
#endif
//@}

/*  Declare or define an instance of the reducer type and its functions for each 
 *  numeric type.
 */
CILK_C_REDUCER_OPXOR_INSTANCE(char,                 char)
CILK_C_REDUCER_OPXOR_INSTANCE(unsigned char,        uchar)
CILK_C_REDUCER_OPXOR_INSTANCE(signed char,          schar)
CILK_C_REDUCER_OPXOR_INSTANCE(wchar_t,              wchar_t)
CILK_C_REDUCER_OPXOR_INSTANCE(short,                short)
CILK_C_REDUCER_OPXOR_INSTANCE(unsigned short,       ushort)
CILK_C_REDUCER_OPXOR_INSTANCE(int,                  int)
CILK_C_REDUCER_OPXOR_INSTANCE(unsigned int,         uint)
CILK_C_REDUCER_OPXOR_INSTANCE(unsigned int,         unsigned) /* alternate name */
CILK_C_REDUCER_OPXOR_INSTANCE(long,                 long)
CILK_C_REDUCER_OPXOR_INSTANCE(unsigned long,        ulong)
CILK_C_REDUCER_OPXOR_INSTANCE(long long,            longlong)
CILK_C_REDUCER_OPXOR_INSTANCE(unsigned long long,   ulonglong)

//@endcond

__CILKRTS_END_EXTERN_C

//@}

//@}

#endif /*  REDUCER_OPXOR_H_INCLUDED */
