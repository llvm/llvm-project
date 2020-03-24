/*  reducer.h                  -*- C++ -*-
 *
 *  Copyright (C) 2009-2018, Intel Corporation
 *  All rights reserved.
 *  
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
 *  
 *  *********************************************************************
 *  
 *  PLEASE NOTE: This file is a downstream copy of a file maintained in
 *  a repository at cilkplus.org. Changes made to this file that are not
 *  submitted through the contribution process detailed at
 *  http://www.cilkplus.org/submit-cilk-contribution will be lost the next
 *  time that a new version is released. Changes only submitted to the
 *  GNU compiler collection or posted to the git repository at
 *  https://bitbucket.org/intelcilkruntime/intel-cilk-runtime are
 *  not tracked.
 *  
 *  We welcome your contributions to this open source project. Thank you
 *  for your assistance in helping us improve Cilk Plus.
 */

/** @file reducer.h
 *
 *  @brief Defines foundation classes for creating Intel(R) Cilk(TM) Plus reducers.
 *
 *  @ingroup Reducers
 *
 *  @see @ref pagereducers
 *
 *  @defgroup Reducers Reducers
 */

#ifndef REDUCER_H_INCLUDED
#define REDUCER_H_INCLUDED

#include "cilk/hyperobject_base.h"
#include "cilk/metaprogramming.h"

#ifdef __cplusplus

//===================== C++ interfaces ===================================

#include <new>

namespace cilk {

/** Class for provisionally constructed objects.
 *
 *  The monoid_base<T,V>::construct() functions manually construct both a
 *  monoid and a view. If one of these is constructed successfully, and the
 *  construction of the other (or some other initialization) fails, then the
 *  first one must be destroyed to avoid a memory leak. Because the
 *  construction is explicit, the destruction must be explicit, too.
 *
 *  A provisional_guard object wraps a pointer to a newly constructed
 *  object. A call to its confirm() function confirms that the object is
 *  really going to be used. If the guard is destroyed without being
 *  confirmed, then the pointed-to object is destroyed (but not
 *  deallocated).
 *
 *  Expected usage:
 *
 *      provisional_guard<T1> x1_provisional( new (x1) T1 );
 *      … more initialization …
 *      x1_provisional.confirm();
 *
 *  or
 *
 *      provisional_guard<T1> x1_provisional( new (x1) T1 );
 *      x1_provisional.confirm_if( new (x2) T2 );
 *
 *  If an exception is thrown in the "more initialization" code in the
 *  first example, or in the `T2` constructor in the second example, then
 *  `x1_provisional` will not be confirmed, so when its destructor is
 *  called during exception unwinding, the `T1` object that was constructed
 *  in `x1` will be destroyed.
 *
 *  **NOTE**: Do *not* be tempted to chain a `provisional_guard`
 *  constructor with `confirm_if` as in this example: 
 *
 *      // BAD IDEA
 *      provisional_guard<T1>( new (x1) T1 ).confirm_if( new (x2) T2 );
 *
 *  The code above is problematic because the evaluation of the T2
 *  constructor is unsequenced with respect to the call to the
 *  `provisional_guard` constructor (and with respect the T1 constructor).
 *  Thus, the compiler may choose to evaluate `new (x2) T2` before
 *  constructing the guard and leak the T1 object if the `T2` constructor
 *  throws.
 *
 *  @tparam Type    The type of the provisionally constructed object.
 */
template <typename Type>
class provisional_guard {
    Type* m_ptr;

public:

    /** Constructor. Creates a guard for a provisionally constructed object.
     *
     *  @param ptr  A pointer to the provisionally constructed object.
     */
    provisional_guard(Type* ptr) : m_ptr(ptr) {}

    /** Destructor. Destroy the object pointed to by the contained pointer
     *  if it has not been confirmed.
     */
    ~provisional_guard() { if (m_ptr) m_ptr->~Type(); }

    /** Confirm the provisional construction. Do *not* delete the contained
     *  pointer when the guard is destroyed.
     */
    void confirm() { m_ptr = 0; }

    /** Confirm provisional construction if argument is non-null. Note that
     *  if an exception is thrown during evaluation of the argument
     *  expression, then this function will not be called, and the
     *  provisional object will not be confirmed. This allows the usage:
     *
     *      x1_provisional.confirm_if( new (x2) T2() );
     *
     *  @param cond An arbitrary pointer. The provisional object will be
     *              confirmed if @a cond is not null.
     *
     *  @returns    The value of the @a cond argument.
     */
    template <typename Cond>
    Cond* confirm_if(Cond* cond) { if (cond) m_ptr = 0; return cond; }
};

/** Base class for defining monoids.
 *
 *  The monoid_base class template is useful for creating classes that model
 *  the monoid concept. It provides the core type and memory management
 *  functionality.  A subclass of monoid_base need only declare and implement
 *  the `identity` and `reduce` functions.
 *
 *  The monoid_base class also manages the integration between the monoid, the
 *  reducer class that is based on it, and an optional view class which wraps
 *  value objects and restricts access to their operations.
 *
 *  @tparam Value   The value type for the monoid.
 *  @tparam View    An optional view class that serves as a proxy for the value
 *                  type.
 *
 *  @see monoid_with_view
 */
template <typename Value, typename View = Value>
class monoid_base
{

public:

    /** Value type of the monoid.
     */
    typedef Value   value_type;

    /** View type of the monoid. Defaults to be the same as the value type.
     *  @see monoid_with_view
     */
    typedef View    view_type;

    enum {
        /** Should reducers created with this monoid be aligned?
         *
         *  @details
         *  "Aligned" means that the view is allocated at a cache-line aligned
         *  offset in the reducer, and the reducer must be cache-line aligned.
         *  "Unaligned" means that the reducer as a whole is just naturally
         *  aligned, but it contains a large enough block of uninitialized
         *  storage for a cache-line aligned view to be allocated in it at
         *  reducer construction time.
         *
         *  Since the standard heap allocator (new reducer) does not allocate
         *  cache-line aligned storage, only unaligned reducers can be safely
         *  allocated on the heap.
         *
         *  Default is false (unaligned) unless overridden in a subclass.
         *
         *  @since 1.02
         *  (In Intel Cilk Plus library versions 1.0 and 1.01, the default was true.
         *  In Intel Cilk Plus library versions prior to 1.0, reducers were always
         *  aligned, and this data member did not exist.)
         */
        align_reducer = false
    };

    /** Destroys a view. Destroys (without deallocating) the @a View object
     *  pointed to by @a p.
     *
     *  @param p    The address of the @a View object to be destroyed.
     */
    void destroy(view_type* p) const { p->~view_type(); }

    /** Allocates raw memory. Allocate @a s bytes of memory with no
     *  initialization.
     *
     *  @param s    The number of bytes of memory to allocate.
     *  @return     An untyped pointer to the allocated memory.
     */
    void* allocate(size_t s) const { return operator new(s); }

    /** Deallocates raw memory pointed to by @a p
     *  without doing any destruction.
     *
     *  @param p    Pointer to the memory to be deallocated.
     *
     *  @pre        @a p points to a block of memory that was allocated by a
     *              call to allocate().
     */
    void deallocate(void* p) const { operator delete(p); }

    /** Creates the identity value. Constructs (without allocating) a @a View
     *  object representing the default value of the @a Value type.
     *
     *  @param p    A pointer to a block of raw memory large enough to hold a
     *              @a View object.
     *
     *  @post       The memory pointed to by @a p contains a @a View object that
     *              represents the default value of the @a View type.
     *
     *  @deprecated This function constructs the @a View object with its default
     *              constructor, which will often, but not always, yield the
     *              appropriate identity value. Monoid classes should declare
     *              their identity function explicitly, rather than relying on
     *              this default definition.
     */
    void identity(View* p) const { new ((void*) p) View(); }


    /** @name Constructs the monoid and the view with arbitrary arguments.
     *
     *  A @ref reducer object contains monoid and view data members, which are
     *  declared as raw storage (byte arrays), so that they are not implicitly
     *  constructed when the reducer is constructed. Instead, a reducer
     *  constructor calls one of the monoid class's static construct()
     *  functions with the addresses of the monoid and the view, and the
     *  construct() function uses placement `new` to construct them.
     *  This allows the monoid to determine the order in which the monoid and
     *  view are constructed, and to make one of them dependent on the other.
     *
     *  Any arguments to the reducer constructor are just passed on as
     *  additional arguments to the construct() function (after the monoid
     *  and view addresses are set).
     *
     *  A monoid whose needs are satisfied by the suite of construct()
     *  functions below, such as @ref monoid_with_view, can just inherit them
     *  from monoid_base. Other monoids will need to provide their own versions
     *  to override the monoid_base functions.
     */
    //@{

    /** Default-constructs the monoid, identity-constructs the view.
     *
     *  @param monoid   Address of uninitialized monoid object.
     *  @param view     Address of uninitialized initial view object.
     */
    //@{
    template <typename Monoid>
    static void construct(Monoid* monoid, View* view)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid() );
        monoid->identity(view);
        guard.confirm();
    }
    //@}

    /** Default-constructs the monoid, and passes one to five const reference
     *  arguments to the view constructor.
     */
    //@{

    template <typename Monoid, typename T1>
    static void construct(Monoid* monoid, View* view, const T1& x1)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid() );
        guard.confirm_if( new((void*) view) View(x1) );
    }

    template <typename Monoid, typename T1, typename T2>
    static void construct(Monoid* monoid, View* view,
                            const T1& x1, const T2& x2)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid() );
        guard.confirm_if( new((void*) view) View(x1, x2) );
    }

    template <typename Monoid, typename T1, typename T2, typename T3>
    static void construct(Monoid* monoid, View* view,
                            const T1& x1, const T2& x2, const T3& x3)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid() );
        guard.confirm_if( new((void*) view) View(x1, x2, x3) );
    }

    template <typename Monoid, typename T1, typename T2, typename T3,
                typename T4>
    static void construct(Monoid* monoid, View* view,
                            const T1& x1, const T2& x2, const T3& x3,
                            const T4& x4)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid() );
        guard.confirm_if( new((void*) view) View(x1, x2, x3, x4) );
    }

    template <typename Monoid, typename T1, typename T2, typename T3,
                typename T4, typename T5>
    static void construct(Monoid* monoid, View* view,
                            const T1& x1, const T2& x2, const T3& x3,
                            const T4& x4, const T5& x5)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid() );
        guard.confirm_if( new((void*) view) View(x1, x2, x3, x4, x5) );
    }

    //@}

    /** Default-constructs the monoid, and passes one non-const reference
     *  argument to the view constructor.
     */
    //@{
    template <typename Monoid, typename T1>
    static void construct(Monoid* monoid, View* view, T1& x1)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid() );
        guard.confirm_if( new((void*) view) View(x1) );
    }
    //@}

    /** Copy-constructs the monoid, and identity-constructs the view
     *  constructor.
     *
     *  @param monoid   Address of uninitialized monoid object.
     *  @param view     Address of uninitialized initial view object.
     *  @param m        Object to be copied into `*monoid`
     */
    //@{
    template <typename Monoid>
    static void construct(Monoid* monoid, View* view, const Monoid& m)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid(m) );
        monoid->identity(view);
        guard.confirm();
    }
    //@}

    /** Copy-constructs the monoid, and passes one to four const reference
     *  arguments to the view constructor.
     */
    //@{

    template <typename Monoid, typename T1>
    static void construct(Monoid* monoid, View* view, const Monoid& m,
                            const T1& x1)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid(m) );
        guard.confirm_if( new((void*) view) View(x1) );
    }

    template <typename Monoid, typename T1, typename T2>
    static void construct(Monoid* monoid, View* view, const Monoid& m,
                            const T1& x1, const T2& x2)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid(m) );
        guard.confirm_if( new((void*) view) View(x1, x2) );
    }

    template <typename Monoid, typename T1, typename T2, typename T3>
    static void construct(Monoid* monoid, View* view, const Monoid& m,
                            const T1& x1, const T2& x2, const T3& x3)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid(m) );
        guard.confirm_if( new((void*) view) View(x1, x2, x3) );
    }

    template <typename Monoid, typename T1, typename T2, typename T3,
                typename T4>
    static void construct(Monoid* monoid, View* view, const Monoid& m,
                            const T1& x1, const T2& x2, const T3& x3,
                            const T4& x4)
    {
        provisional_guard<Monoid> guard( new((void*) monoid) Monoid(m) );
        guard.confirm_if( new((void*) view) View(x1, x2, x3, x4) );
    }

    //@}

    //@}
};


/** Monoid class that gets its value type and identity and reduce operations
 *  from its view.
 *
 *  A simple implementation of the monoid-view-reducer architecture would
 *  distribute knowledge about the type and operations for the reduction
 *  between the monoid and the view - the identity and reduction operations are
 *  specified in the monoid, the reduction operations are implemented in the
 *  view, and the value type is specified in both the monoid and the view.
 *  This is inelegant.
 *
 *  monoid_with_view is a subclass of @ref monoid_base that gets its value type
 *  and its identity and reduction operations from its view class. No
 *  customization of the monoid_with_view class itself is needed beyond
 *  instantiating it with an appropriate view class. (Customized subclasses of
 *  monoid_with_view may be needed for other reasons, such as to keep some
 *   state for the reducer.) All of the Intel Cilk Plus predefined reducers use
 *  monoid_with_view or one of its subclasses.
 *
 *  The view class `View` of a monoid_with_view must provide the following
 *  public definitions:
 *
 *  Definition                       | Meaning
 *  ---------------------------------|--------
 *  `value_type`                     | a typedef of the value type for the reduction
 *  `View()`                         | a default constructor which constructs the identity value for the reduction
 *  `void reduce(const View* other)` | a member function which applies the reduction operation to the values of `this` view and the `other` view, leaving the result as the value of `this` view, and leaving the value of the `other` view undefined (but valid)
 *
 *  @tparam View    The view class for the monoid.
 *  @tparam Align   If true, reducers instantiated on this monoid will be
 *                  cache-aligned. By default, library reducers (unlike legacy
 *                  library reducer _wrappers_) are aligned only as required by
 *                  contents.
 */
template <class View, bool Align = false>
class monoid_with_view : public monoid_base<typename View::value_type, View>
{
public:
    /** Should reducers created with this monoid be aligned?
     */
    enum { align_reducer = Align };

    /** Create the identity value.
     *
     *  Implements the monoid `identity` operation by using the @a View class's
     *  default constructor.
     *
     *  @param  p   A pointer to a block of raw memory large enough to hold a
     *              @p View object.
     */
    void identity(View* p) const { new((void*) p) View(); }

    /** Reduce the values of two views.
     *
     *  Implements the monoid `reduce` operation by calling the left view's
     *  `%reduce()` function with the right view as an operand.
     *
     *  @param  left    The left operand of the reduce operation.
     *  @param  right   The right operand of the reduce operation.
     *  @post           The left view contains the result of the reduce
     *                  operation, and the right view is undefined.
     */
    void reduce(View* left, View* right) const { left->reduce(right); }
};


/** Base class for simple views with (usually) scalar values.
 *
 *  The scalar_view class is intended as a base class which provides about half
 *  of the required definitions for simple views. It defines the `value_type`
 *  required by a @ref monoid_with_view (but not the identity constructor and
 *  reduce operation, which are inherently specific to a particular kind of
 *  reduction). It also defines the value access functions which will be called
 *  by the corresponding @ref reducer functions. (It uses copy semantics for
 *  the view_move_in() and view_move_out() functions, which is appropriate
 *  for simple scalar types, but not necessarily for more complex types like
 *  STL containers.
 *
 *  @tparam Type    The type of value wrapped by the view.
 */
template <typename Type>
class scalar_view
{
protected:
    Type m_value;       ///< The wrapped accumulator variable.

public:
    /** Value type definition required by @ref monoid_with_view.
     */
    typedef Type value_type;

    /** Default constructor.
     */
    scalar_view() : m_value() {}

    /** Value constructor.
     */
    scalar_view(const Type& v) : m_value(v) {}

    /** @name Value functions required by the reducer class.
     *
     *  Note that the move in/out functions use simple assignment semantics.
     */
    //@{

    /** Set the value of the view.
     */
    void view_move_in(Type& v) { m_value = v; }

    /** Get the value of the view.
     */
    void view_move_out(Type& v) { v = m_value; }

    /** Set the value of the view.
     */
    void view_set_value(const Type& v) { m_value = v; }

    /** Get the value of the view.
     */
    Type const& view_get_value() const { return m_value; }

    /** Type returned by view_get_value.
     */
    typedef Type const& return_type_for_get_value;

    /** Get a reference to the value contained in the view. For legacy
     *  reducer support only.
     */
    Type      & view_get_reference()       { return m_value; }

    /** Get a reference to the value contained in the view. For legacy
     *  reducer support only.
     */
    Type const& view_get_reference() const { return m_value; }
    //@}
};


/** Wrapper class for move-in construction.
 *
 *  Some types allow their values to be _moved_ as an alternative to copying.
 *  Moving a value may be much faster than copying it, but may leave the value
 *  of the move's source undefined. Consider the `swap` operation provided by
 *  many STL container classes:
 *
 *      list<T> x, y;
 *      x = y;      // Copy
 *      x.swap(y);  // Move
 *
 *  The assignment _copies_ the value of `y` into `x` in time linear in the
 *  size of `y`, leaving `y` unchanged. The `swap` _moves_ the  value of `y`
 *  into `x` in constant time, but it also moves the value of `x` into `y`,
 *  potentially leaving `y` undefined.
 *
 *  A move_in_wrapper simply wraps a pointer to an object. It is created by a
 *  call to cilk::move_in(). Passing a move_in_wrapper to a view constructor
 *  (actually, passing it to a reducer constructor, which passes it to the
 *  monoid `construct()` function, which passes it to the view constructor)
 *  allows, but does not require, the value pointed to by the wrapper to be
 *  moved into the view instead of copied.
 *
 *  A view class exercises this option by defining a _move-in constructor_,
 *  i.e., a constructor with a move_in_wrapper parameter. The constructor calls
 *  the wrapper's `value()` function to get a reference to its pointed-to
 *  value, and can then use that reference in a move operation.
 *
 *  A move_in_wrapper also has an implicit conversion to its pointed-to value,
 *  so if a view class does not define a move-in constructor, its ordinary
 *  value constructor will be called with the wrapped value. For example, an
 *  @ref ReducersAdd "op_add" view does not have a move-in constructor, so
 *
 *      int x;
 *      reducer< op_add<int> > xr(move_in(x));
 *
 *  will simply call the `op_add_view(const int &)` constructor. But an
 *  @ref ReducersList "op_list_append" view does have a move-in  constructor,
 *  so
 *
 *      list<int> x;
 *      reducer< op_list_append<int> > xr(move_in(x));
 *
 *  will call the `op_list_append_view(move_in_wrapper<int>)` constructor,
 *  which can `swap` the value of `x` into the view.
 *
 *  @note   Remember that passing the value of a variable to a reducer
 *          constructor using a move_in_wrapper leaves the variable undefined.
 *          You cannot assume that the constructor either will or will not copy
 *          or move the value.
 *
 *  @tparam Type    The type of the wrapped value.
 *
 *  @see cilk::move_in()
 */
template <typename Type>
class move_in_wrapper
{
    Type *m_pointer;
public:

    /** Constructor that captures the address of its argument. This is almost
     *  always called from the @ref move_in function.
     */
    explicit move_in_wrapper(Type& ref) : m_pointer(&ref) { }

    /** Implicit conversion to the wrapped value. This allows a move_in_wrapper
     *  to be used where a value of the wrapped type is expected, in which case
     *  the wrapper is completely transparent.
     */
    operator Type&() const { return *m_pointer; }

    /** Get a reference to the pointed-to value. This has the same effect as
     *  the implicit conversion, but makes the intent clearer in a move-in
     *  constructor.
     */
    Type& value() const { return *m_pointer; }
};

/** Function to create a move_in_wrapper for a value.
 *
 *  @tparam Type    The type of the argument, which will be the `type` of the
 *                  created wrapper.
 *
 *  @see move_in_wrapper
 */
template <typename Type>
inline
move_in_wrapper<Type> move_in(Type& ref)
    { return move_in_wrapper<Type>(ref); }


/** @copydoc move_in(Type&)
 *
 *  @note   Applying a function that is explicitly specified as modifying its
 *          argument to a const argument is obviously an irrational thing to
 *          do. This move_in() variant is just provided to allow calling a
 *          move-in constructor with a function return value, which the
 *          language treats as a const. Using it for any other purpose will
 *          probably end in tears.
 */
template <typename Type>
inline
move_in_wrapper<Type> move_in(const Type& ref)
    { return move_in_wrapper<Type>(ref); }


/** Wrapper class to allow implicit downcasts to reducer subclasses.
 *
 *  The Intel Cilk Plus library contains a collection of reducer wrapper classes which
 *  were created before the `cilk::reducer<Monoid>` style was developed. For
 *  example, `cilk::reducer_opadd<Type>` provided essentially the same
 *  functionality that is now provided by
 *  `cilk::reducer< cilk::op_add<Type> >`. These legacy reducer classes are
 *  deprecated, but still supported, and they have been reimplemented as
 *  subclasses of the corresponding `cilk::reducer` classes. For example:
 *
 *      template <class T>
 *      reducer_opadd<T> : public reducer< op_add<T> > { ... };
 *
 *  This reimplementation allows transparent conversion between legacy and
 *  new reducers. That is, a `reducer<op_add>*` or `reducer<op_add>&` can be
 *  used anywhere that a `reducer_opadd*` or `reducer_opadd&` is expected,
 *  and vice versa.
 *
 *  The conversion from the legacy reducer to the new reducer is just an
 *  up-cast, which is provided for free by C++. The conversion from the new
 *  reducer to the legacy reducer is a down-cast, though, which requires an
 *  explicit conversion member function in the `reducer` class. The challenge
 *  is to define a function in the reducer template class which will convert
 *  each cilk::reducer specialization to the corresponding legacy reducer,
 *  if there is one.
 *
 *  The trick is in the legacy_reducer_downcast template class, which provides
 *  a mapping from  `cilk::reducer` specializations to legacy reducer classes.
 *  `reducer<Monoid>` has a conversion function to convert itself to
 *  `legacy_reducer_downcast< reducer<Monoid> >::%type`. By default,
 *  `legacy_reducer_downcast<Reducer>::%type` is just a trivial subclass of
 *  `Reducer`, which is uninteresting, but a reducer with a legacy counterpart
 *  will have a specialization of `legacy_reducer_downcast` whose `type` is
 *  the corresponding legacy reducer. For example:
 *
 *      template <typename Type>
 *      struct legacy_reducer_downcast< reducer< op_add<Type> > >
 *      {
 *          typedef reducer_opadd<Type> type;
 *      };
 *
 *
 *  @tparam Reducer The new-style reducer class whose corresponding legacy
 *                  reducer class is `type`, if there is such a legacy reducer
 *                  class.
 */
template <typename Reducer>
struct legacy_reducer_downcast
{
    /** The related legacy reducer class.
     *
     *  By default, this is just a trivial subclass of Reducer, but it can be
     *  overridden in the specialization of legacy_reducer_downcast for
     *  a reducer that has a corresponding legacy reducers.
     */
    struct type : Reducer { };
};


namespace internal {
/// @cond internal

template <typename Value, typename View>
struct reducer_set_get
{
    // sizeof(notchar) != sizeof(char)
    struct notchar { char x[2]; };

    // `does_view_define_return_type_for_get_value(View*)` returns `char` if
    // `View` defines `return_type_for_get_value`, and `notchar` if it doesn't.

    template <typename T>
    struct using_type {};

    template <typename T>
    static char does_view_define_return_type_for_get_value(
                        using_type<typename T::return_type_for_get_value>*);

    template <typename T>
    static notchar does_view_define_return_type_for_get_value(...);

    // `VIEW_DOES_DEFINE_RETURN_TYPE_FOR_GET_VALUE` is true if `View` defines
    // `return_type_for_get_value`.

    enum { VIEW_DOES_DEFINE_RETURN_TYPE_FOR_GET_VALUE =
            sizeof( does_view_define_return_type_for_get_value<View>(0) )
            == sizeof(char) } ;

    // `return_type_for_get_value` is `View::return_type_for_get_value`
    // if it is defined, and just `Value` otherwise.

    template <typename InnerView, bool ViewDoesDefineReturnTypeForGetValue>
    struct return_type_for_view_get_value {
        typedef Value type;
    };

    template <typename InnerView>
    struct return_type_for_view_get_value<InnerView, true> {
        typedef typename InnerView::return_type_for_get_value type;
    };

public:

    typedef
        typename
            return_type_for_view_get_value<
                View,
                VIEW_DOES_DEFINE_RETURN_TYPE_FOR_GET_VALUE
            >::type
        return_type_for_get_value;

    static void move_in(View& view, Value& v)   { view.view_move_in(v); }
    static void move_out(View& view,  Value& v) { view.view_move_out(v); }

    static void set_value(View& view, const Value& v)
        { view.view_set_value(v); }

    static return_type_for_get_value get_value(const View& view)
        { return view.view_get_value(); }
};

template <typename Value>
struct reducer_set_get<Value, Value>
{
    typedef const Value& return_type_for_get_value;

    static void move_in(Value& view, Value& v)   { view = v; }
    static void move_out(Value& view,  Value& v) { v = view; }

    static void set_value(Value& view, const Value& v)
        { view = v; }

    static return_type_for_get_value get_value(const Value& view)
        { return view; }
};

/// @endcond


/** Base class defining the data layout that is common to all reducers.
 */
template <typename Monoid>
class reducer_base {
    typedef typename Monoid::view_type view_type;

    // This makes the reducer a hyper-object. (Partially initialized in
    // the derived reducer_content class.)
    //
    __cilkrts_hyperobject_base      m_base;

    // The monoid is allocated here as raw bytes, and is constructed explicitly
    // by a call to the monoid_type::construct() function in the constructor of
    // the `reducer` subclass.
    //
    storage_for_object<Monoid>      m_monoid;

    // Used for sanity checking at destruction.
    //
    void*                           m_initialThis;

    // The leftmost view comes next. It is defined in the derived
    // reducer_content class.

    /** @name C-callable wrappers for the C++-coded monoid dispatch functions.
     */
    //@{

    static void reduce_wrapper(void* r, void* lhs, void* rhs);
    static void identity_wrapper(void* r, void* view);
    static void destroy_wrapper(void* r, void* view);
    static void* allocate_wrapper(void* r, __STDNS size_t bytes);
    static void deallocate_wrapper(void* r, void* view);

    //@}

protected:

    /** Constructor.
     *
     *  @param  leftmost    The address of the leftmost view in the reducer.
     */
    reducer_base(char* leftmost)
    {
        static const cilk_c_monoid c_monoid_initializer = {
            (cilk_c_reducer_reduce_fn_t)     &reduce_wrapper,
            (cilk_c_reducer_identity_fn_t)   &identity_wrapper,
            (cilk_c_reducer_destroy_fn_t)    &destroy_wrapper,
            (cilk_c_reducer_allocate_fn_t)   &allocate_wrapper,
            (cilk_c_reducer_deallocate_fn_t) &deallocate_wrapper
        };

        m_base.__c_monoid = c_monoid_initializer;
        m_base.__flags = 0;
        m_base.__view_offset = (char*)leftmost - (char*)this;
        m_base.__view_size = sizeof(view_type);
        m_initialThis = this;

        __cilkrts_hyper_create(&m_base);
    }

    /** Destructor.
     */
    __CILKRTS_STRAND_STALE(~reducer_base())
    {
        // Make sure we haven't been memcopy'd or corrupted
        __CILKRTS_ASSERT(
            this == m_initialThis ||
            // Allow for a layout bug that may put the initialThis field one
            // word later in 1.0 reducers than in 0.9  and 1.1 reducers.
            this == *(&m_initialThis + 1)
        );
        __cilkrts_hyper_destroy(&m_base);
    }

    /** Monoid data member.
     *
     *  @return A pointer to the reducer's monoid data member.
     */
    Monoid* monoid_ptr() { return &m_monoid.object(); }

    /** Leftmost view data member.
     *
     *  @return A pointer to the reducer's leftmost view data member.
     *
     *  @note   This function returns the address of the *leftmost* view,
     *          which is unique for the lifetime of the reducer. It is
     *          intended to be used in constructors and destructors.
     *          Use the reducer::view() function to access the per-strand
     *          view instance.
     */
    view_type* leftmost_ptr()
    {
        char* view_addr = (char*)this + m_base.__view_offset;
        return reinterpret_cast<view_type*>(view_addr);
    }

public:

    /** @name Access the current view.
     *
     *  These functions return a reference to the instance of the reducer's
     *  view that was created for the current strand of a parallel computation
     *  (and create it if it doesn't already exist). Note the difference from
     *  the (private) leftmost_ptr() function, which returns a pointer to the
     *  _leftmost_ view, which is the same in all strands.
     */
    //@{

    /** Per-strand view instance.
     *
     *  @return A reference to the per-strand view instance.
     */
    view_type& view()
    {
        return *static_cast<view_type *>(__cilkrts_hyper_lookup(&m_base));
    }

    /** @copydoc view()
     */
    const view_type& view() const
    {
        return const_cast<reducer_base*>(this)->view();
    }

    //@}

    /** Initial view pointer field.
     *
     *  @internal
     *
     *  @return a reference to the m_initialThis field.
     *
     *  @note   This function is provided for "white-box" testing of the
     *          reducer layout code. There is never any reason for user code
     *          to call it.
     */
    const void* const & initial_this() const { return m_initialThis; }
};

template <typename Monoid>
void reducer_base<Monoid>::reduce_wrapper(void* r, void* lhs, void* rhs)
{
    Monoid* monoid = static_cast<reducer_base*>(r)->monoid_ptr();
    monoid->reduce(static_cast<view_type*>(lhs),
                         static_cast<view_type*>(rhs));
}

template <typename Monoid>
void reducer_base<Monoid>::identity_wrapper(void* r, void* view)
{
    Monoid* monoid = static_cast<reducer_base*>(r)->monoid_ptr();
    monoid->identity(static_cast<view_type*>(view));
}

template <typename Monoid>
void reducer_base<Monoid>::destroy_wrapper(void* r, void* view)
{
    Monoid* monoid = static_cast<reducer_base*>(r)->monoid_ptr();
    monoid->destroy(static_cast<view_type*>(view));
}

template <typename Monoid>
void* reducer_base<Monoid>::allocate_wrapper(void* r, __STDNS size_t bytes)
{
    Monoid* monoid = static_cast<reducer_base*>(r)->monoid_ptr();
    return monoid->allocate(bytes);
}

template <typename Monoid>
void reducer_base<Monoid>::deallocate_wrapper(void* r, void* view)
{
    Monoid* monoid = static_cast<reducer_base*>(r)->monoid_ptr();
    monoid->deallocate(static_cast<view_type*>(view));
}


/** Base class defining the data members of a reducer.
 *
 *  @tparam Aligned The `m_view` data member, and therefore the entire
 *                  structure, are cache-line aligned if this parameter
 *                  is `true'.
 */
template <typename Monoid, bool Aligned = Monoid::align_reducer>
class reducer_content;

/** Base class defining the data members of an aligned reducer.
 */
template <typename Monoid>
class reducer_content<Monoid, true> : public reducer_base<Monoid>
{
    typedef typename Monoid::view_type view_type;

    // The leftmost view is defined as raw bytes. It will be constructed
    // by the monoid `construct` function. It is cache-aligned, which
    // will push it into a new cache line. Furthermore, its alignment causes
    // the reducer as a whole to be cache-aligned, which makes the reducer
    // size a multiple of a cache line. Since there is nothing in the reducer
    // after the view, all this means that the leftmost view gets one or more
    // cache lines all to itself, which prevents false sharing.
    //
    __CILKRTS_CACHE_ALIGN
    char m_leftmost[sizeof(view_type)];

    /** Test if the reducer is cache-line-aligned.
     *
     *  Used in assertions.
     */
    bool reducer_is_cache_aligned() const
        { return 0 == ((std::size_t) this & (__CILKRTS_CACHE_LINE__ - 1)); }

protected:

    /** Constructor.
     */
    reducer_content() : reducer_base<Monoid>((char*)&m_leftmost)
    {
#ifndef CILK_IGNORE_REDUCER_ALIGNMENT
    assert(reducer_is_cache_aligned() &&
           "Reducer should be cache aligned. Please see comments following "
           "this assertion for explanation and fixes.");
#endif
    /*  "REDUCER SHOULD BE CACHE ALIGNED" ASSERTION.
     *
     *  This Reducer class instantiation specifies cache-line alignment of the
     *  leftmost view field (and, implicitly, of the reducer itself). You got
     *  this assertion because a reducer with this class was allocated at a
     *  non-cache-aligned address, probably because it was allocated on the
     *  heap with `new`. This can be a problem for two reasons:
     *
     *  1.  If the leftmost view is not on a cache line by itself, there might
     *      be a slowdown resulting from accesses to the same cache line from
     *      different threads.
     *
     *  2.  The compiler thinks that reducer is cache-line aligned, but it
     *      really isn't. If the reducer is contained in a structure, then the
     *      compiler will believe that the containing structure, and other
     *      fields contained in it, are also more aligned than they really
     *      are. In particular, if the structure contains a numeric array that
     *      is used in a vectorizable loop, then the compiler might generate
     *      invalid vector instructions, resulting in a runtime error.
     *
     *  The compiler will always allocate reducer variables, and structure
     *  variables containing reducers, with their required alignment.
     *  Reducers, and structures containing a reducer, which are allocated
     *  on the heap with `new` will _not_ be properly aligned.
     *
     *  There are three ways that you can fix this assertion failure.
     *
     *  A.  Rewrite your code to use the new-style `reducer< op_XXX<Type> >`
     *      instead of the legacy `reducer_XXX<type>`. The new-style reducers
     *      are not declared to be cache-aligned, and will work properly if
     *      they are not cache-aligned.
     *
     *  B.  If you must allocate an old-style reducer or a structure containing
     *      a reducer on the heap, figure out how to align it correctly. The
     *      suggested fix is to use `cilk::aligned_new()` and
     *      `cilk::aligned_delete()` instead of `new` and `delete`, as follows:
     *
     *          Type* ptr = cilk::aligned_new<Type>(constructor-arguments);
     *          cilk::aligned_delete(ptr);
     *
     *  C.  Define the macro CILK_IGNORE_REDUCER_ALIGNMENT, which will suppress
     *      the assertion check. Do this only if you are comfortable that
     *      problem (2) above will not occur.
     */
    }
};

/** Base class defining the data members of an unaligned reducer.
 */
template <typename Monoid>
class reducer_content<Monoid, false> : public reducer_base<Monoid>
{
    typedef typename Monoid::view_type view_type;      ///< The view type.

    // Reserve space for the leftmost view. The view will be allocated at an
    // aligned offset in this space at runtime, to guarantee that the view
    // will get one or more cache lines all to itself, to prevent false
    // sharing.
    //
    // The number of bytes to reserve is determined as follows:
    // * Start with the view size.
    // * Round up to a multiple of the cache line size, to get the total size
    //   of the cache lines that will be dedicated to the view.
    // * Add (cache line size - 1) filler bytes to guarantee that the reserved
    //   area will contain a cache-aligned block of the required cache lines,
    //   no matter where the reserved area starts.
    //
    char m_leftmost[
        // View size rounded up to multiple cache lines
        (   (sizeof(view_type) + __CILKRTS_CACHE_LINE__ - 1)
            & ~ (__CILKRTS_CACHE_LINE__ - 1)
        )
        // plus filler to allow alignment.
        + __CILKRTS_CACHE_LINE__ - 1
        ];

protected:

    /** Constructor. Find the first cache-aligned position in the reserved
     *  area, and pass it to the base constructor as the leftmost view
     *  address.
     */
    reducer_content() :
        reducer_base<Monoid>(
            (char*)( ((std::size_t)&m_leftmost + __CILKRTS_CACHE_LINE__ - 1)
                     & ~ (__CILKRTS_CACHE_LINE__ - 1) ) )
    {}
};


} // namespace internal


// The __cilkrts_hyperobject_ functions are defined differently depending on
// whether a file is compiled with or without the CILK_STUB option. Therefore,
// reducers compiled in the two modes should be link-time incompatible, so that
// object files compiled with stubbed reducers won't be linked into an
// unstubbed program, or vice versa. We achieve this by putting the reducer
// class definition into the cilk::stub namespace in a stubbed compilation.

#ifdef CILK_STUB
namespace stub {
#endif

/** Reducer class.
 *
 *  A reducer is instantiated on a Monoid.  The Monoid provides the value
 *  type, associative reduce function, and identity for the reducer.
 *
 *  @tparam Monoid  The monoid class that the reducer is instantiated on. It
 *                  must model the @ref reducers_monoid_concept "monoid
 *                  concept".
 *
 *  @see @ref pagereducers
 */
template <class Monoid>
class reducer : public internal::reducer_content<Monoid>
{
    typedef internal::reducer_content<Monoid> base;
    using base::monoid_ptr;
    using base::leftmost_ptr;
  public:
    typedef Monoid                          monoid_type;  ///< The monoid type.
    typedef typename Monoid::value_type     value_type;   ///< The value type.
    typedef typename Monoid::view_type      view_type;    ///< The view type.

  private:
    typedef internal::reducer_set_get<value_type, view_type> set_get;

    reducer(const reducer&);                ///< Disallow copying.
    reducer& operator=(const reducer&);     ///< Disallow assignment.

  public:

    /** @name Constructors
     *
     *  All reducer constructors call the static `construct()` function of the
     *  monoid class to construct the reducer's monoid and leftmost view.
     *
     *  The reducer constructor arguments are simply passed through to the
     *  construct() function.  Thus, the constructor parameters accepted by a
     *  particular reducer class are determined by its monoid class.
     */
    //@{

    /** 0 – 6 const reference parameters.
     */
    //@{

    reducer()
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr());
    }

    template <typename T1>
    reducer(const T1& x1)
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr(), x1);
    }

    template <typename T1, typename T2>
    reducer(const T1& x1, const T2& x2)
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr(), x1, x2);
    }

    template <typename T1, typename T2, typename T3>
    reducer(const T1& x1, const T2& x2, const T3& x3)
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr(), x1, x2, x3);
    }

    template <typename T1, typename T2, typename T3, typename T4>
    reducer(const T1& x1, const T2& x2, const T3& x3, const T4& x4)
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr(), x1, x2, x3, x4);
    }

    template <typename T1, typename T2, typename T3, typename T4, typename T5>
    reducer(const T1& x1, const T2& x2, const T3& x3, const T4& x4,
            const T5& x5)
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr(),
                               x1, x2, x3, x4, x5);
    }

    template <typename T1, typename T2, typename T3, typename T4,
              typename T5, typename T6>
    reducer(const T1& x1, const T2& x2, const T3& x3, const T4& x4,
            const T5& x5, const T6& x6)
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr(),
                               x1, x2, x3, x4, x5, x6);
    }

    //@}

    /** 1 non-const reference parameter.
     */
    //@{

    template <typename T1>
    reducer(T1& x1)
    {
        monoid_type::construct(monoid_ptr(), leftmost_ptr(), x1);
    }

    //@}

    /** Destructor.
     */
    __CILKRTS_STRAND_STALE(~reducer())
    {
        leftmost_ptr()->~view_type();
        monoid_ptr()->~monoid_type();
    }

    //@{
    /** Get the monoid.
     *
     *  @return A reference to the monoid object belonging to this reducer.
     */
    Monoid& monoid() { return *monoid_ptr(); }

    const Monoid& monoid() const
    { return const_cast<reducer*>(this)->monoid(); }
    //@}

    //@{
    /** Access the current view.
     *
     *  Return a reference to the instance of the reducer's view that was
     *  created for the current strand of a parallel computation (and create
     *  it if it doesn't already exist).
     */
          view_type& view()       { return base::view(); }
    const view_type& view() const { return base::view(); }
    //@}


    /** @name Dereference the reducer to get the view.
     *
     *  "Dereferencing" a reducer yields the view for the current strand. The
     *  view, in turn, acts as a proxy for its contained value, exposing only
     *  those operations which are consistent with the reducer's monoid. Thus,
     *  all modifications of the reducer's accumulator variable are written as
     *
     *      *reducer OP ...
     *
     *  or
     *
     *      reducer->func(...)
     *
     *  (The permitted operations on a reducer's accumulator are listed in the
     *  documentation for that particular kind of reducer.)
     *
     *  @note   `*r` is a synonym for `r.view()`. Recommended style is to use
     *          `*r` (or `r->`) in the common case where code is simply
     *          updating the accumulator variable wrapped in the view, and to
     *          use `r.view()` in the unusual case where it is desirable to
     *          call attention to the view itself.
     */
    //@{

    //@{
    /** Dereference operator.
     *
     *  @return A reference to the per-strand view instance.
     */
    view_type&       operator*()       { return view(); }
    view_type const& operator*() const { return view(); }
    //@}

    //@{
    /** Pointer operator.
     *
     *  @return A pointer to the per-strand view instance.
     */
    view_type*       operator->()       { return &view(); }
    view_type const* operator->() const { return &view(); }
    //@}

    //@{
    /** Deprecated view access.
     *
     *  `r()` is a synonym for `*r` which was used with early versions of
     *  Intel Cilk Plus reducers. `*r` is now the preferred usage.
     *
     *  @deprecated Use operator*() instead of operator()().
     *
     *  @return A reference to the per-strand view instance.
     */
    view_type&       operator()()       { return view(); }
    view_type const& operator()() const { return view(); }
    //@}

    //@}

    /** @name Set and get the value.
     *
     *  These functions are used to set an initial value for the reducer before
     *  starting the reduction, or to get the final value after the reduction
     *  is complete.
     *
     *  @note   These functions are completely different from the view
     *          operations that are made available via operator*() and
     *          operator->(), which are used to _modify_ the reducer's value
     *          _during_ the reduction.
     *
     *  @warning    These functions _can_ be called at any time, and in
     *              general, they will refer to the value contained in the view
     *              for the current strand. However, using them other than to
     *              set the reduction's initial value or get its final value
     *              will almost always result in undefined behavior.
     */
    //@{

    /** Move a value into the reducer.
     *
     *  This function is used to set the initial value of the reducer's
     *  accumulator variable by either copying or _moving_ the value of @a obj
     *  into it. Moving a value can often be performed in constant time, even
     *  for large container objects, but has the side effect of leaving the
     *  value of @a obj undefined. (See the description of the
     *  @ref move_in_wrapper class for a discussion of moving values.)
     *
     *  @par    Usage
     *          A move_in() call to initialize a reducer is often paired with a
     *          move_out() call to get its final value:
     *
     *              reducer<Type> xr;
     *              xr.move_in(x);
     *              … do the reduction …
     *              xr.move_out(x);
     *
     *  @par Assumptions
     *      -   You cannot assume either that this will function will copy its
     *          value or that it will move it.
     *      -   You must assume that the value of @a obj will be undefined
     *          after the call to move_in().
     *      -   You can assume that move_in() will be at least as efficient as
     *          set_value(), and you should therefore prefer move_in() unless
     *          you need the value of @a obj to be unchanged after the call.
     *          (But you should usually prefer the move-in constructor over a
     *          move_in() call - see the note below.)
     *
     *  @note   The behavior of a default constructor followed by move-in
     *          initialization:
     *
     *              reducer<Type> xr;
     *              xr.move_in(x);
     *
     *  @note   is not necessarily the same as a move-in constructor:
     *
     *      reducer<Type> xr(move_in(x));
     *
     *  @note   In particular, when @a Type is a container type with a
     *          non-empty allocator, the move-in constructor will create the
     *          accumulator variable with the same allocator as the input
     *          argument @a x, while the default constructor will create the
     *          accumulator variable with a default allocator. The mismatch of
     *          allocators in the latter case means that the input argument
     *          @a x may have to be copied in linear time instead of being
     *          moved in constant time.
     *
     *  @note   Best practice is to prefer the move-in constructor over the
     *          move-in function unless the move-in function is required for
     *          some specific reason.
     *
     *  @warning    Calling this function other than to set the initial value
     *              for a reduction will almost always result in undefined
     *              behavior.
     *
     *  @param  obj The object containing the value that will be moved into the
     *              reducer.
     *
     *  @post   The reducer contains the value that was initially in @a obj.
     *  @post   The value of @a obj is undefined.
     *
     *  @see set_value()
     */
    void move_in(value_type& obj) { set_get::move_in(view(), obj);}

    /** Move the value out of the reducer.
     *
     *  This function is used to retrieve the final value of the reducer's
     *  accumulator variable by either copying or _moving_ the value of @a obj
     *  into it. Moving a value can often be performed in constant time, even
     *  for large container objects, but has the side effect of leaving the
     *  value of the reducer's accumulator variable undefined. (See the
     *  description of the @ref move_in_wrapper class for a discussion of
     *  moving values.)
     *
     *  @par    Usage
     *          A move_in() call to initialize a reducer is often paired with a
     *          move_out() call to get its final value:
     *
     *              reducer<Type> xr;
     *              xr.move_in(x);
     *              … do the reduction …
     *              xr.move_out(x);
     *
     *  @par Assumptions
     *      -   You cannot assume either that this will function will copy its
     *          value or that it will move it.
     *      -   You must assume that the value of the reducer's accumulator
     *          variable will be undefined after the call to move_out().
     *      -   You can assume that move_out() will be at least as efficient as
     *          get_value(), and you should therefore prefer move_out() unless
     *          you need the accumulator variable to be preserved after the
     *          call.
     *
     *  @warning    Calling this function other than to retrieve the final
     *              value of a reduction will almost always result in undefined
     *              behavior.
     *
     *  @param  obj The object that the value of the reducer will be moved into.
     *
     *  @post   @a obj contains the value that was initially in the reducer.
     *  @post   The value of the reducer is undefined.
     *
     *  @see get_value()
     */
    void move_out(value_type& obj) { set_get::move_out(view(), obj); }

    /** Set the value of the reducer.
     *
     *  This function sets the initial value of the reducer's accumulator
     *  variable to the value of @a obj.
     *
     *  @note   The behavior of a default constructor followed by
     *          initialization:
     *
     *      reducer<Type> xr;
     *      xr.set_value(x);
     *
     *  @note   is not necessarily the same as a value constructor:
     *
     *      reducer<Type> xr(x);
     *
     *  @note   In particular, when @a Type is a container type with a
     *          non-empty allocator, the value constructor will create the
     *          accumulator variable with the same allocator as the input
     *          argument @a x, while the default constructor will create the
     *          accumulator variable with a default allocator.
     *
     *  @warning    Calling this function other than to set the initial value
     *              for a reduction will almost always result in undefined
     *              behavior.
     *
     *  @param  obj The object containing the value that will be copied into
     *              the reducer.
     *
     *  @post   The reducer contains a copy of the value in @a obj.
     *
     *  @see move_in()
     */
    void set_value(const value_type& obj) { set_get::set_value(view(), obj); }

    /** Get the value of the reducer.
     *
     *  This function gets the final value of the reducer's accumulator
     *  variable.
     *
     *  @warning    Calling this function other than to retrieve the final
     *              value of a reduction will almost always result in undefined
     *              behavior.
     *
     *  @return     A reference to the value contained in the reducer.
     *
     *  @see move_out()
     */
    typename set_get::return_type_for_get_value get_value() const
        { return set_get::get_value(view()); }

    //@}

    /** Implicit downcast to legacy reducer wrapper, if any.
     *
     *  @see legacy_reducer_downcast
     */
    operator typename legacy_reducer_downcast<reducer>::type& ()
    {
        typedef typename legacy_reducer_downcast<reducer>::type downcast_type;
        return *reinterpret_cast<downcast_type*>(this);
    }


    /** Implicit downcast to legacy reducer wrapper, if any.
     *
     *  @see legacy_reducer_downcast
     */
    operator const typename legacy_reducer_downcast<reducer>::type& () const
    {
        typedef typename legacy_reducer_downcast<reducer>::type downcast_type;
        return *reinterpret_cast<const downcast_type*>(this);
    }
};

#ifdef CILK_STUB
} // namespace stub
using stub::reducer;
#endif

} // end namespace cilk

#endif /* __cplusplus */

/** @page page_reducers_in_c Creating and Using Reducers in C
 *
 *  @tableofcontents
 *
 *  The Intel Cilk Plus runtime supports reducers written in C as well as in C++. The
 *  basic logic is the same, but the implementation details are very
 *  different. The C++ reducer implementation uses templates heavily to create
 *  very generic components. The C reducer implementation uses macros, which
 *  are a much blunter instrument. The most immediate consequence is that the
 *  monoid/view/reducer architecture is mostly implicit rather than explicit
 *  in C reducers.
 *
 *  @section reducers_c_overview Overview of Using Reducers in C
 *
 *  The basic usage pattern for C reducers is:
 *
 *  1.  Create and initialize a reducer object.
 *  2.  Tell the Intel Cilk Plus runtime about the reducer.
 *  3.  Update the value contained in the reducer in a parallel computation.
 *  4.  Tell the Intel Cilk Plus runtime that you are done with the reducer.
 *  5.  Retrieve the value from the reducer.
 *
 *  @subsection reducers_c_creation Creating and Initializing a C Reducer
 *
 *  The basic pattern for creating and initializing a reducer object in C is
 *
 *      CILK_C_DECLARE_REDUCER(value-type) reducer-name =
 *          CILK_C_INIT_REDUCER(value-type,
 *                              reduce-function,
 *                              identity-function,
 *                              destroy-function,
 *                              initial-value);
 *
 *  This is simply an initialized definition of a variable named
 *  _reducer-name_. The @ref CILK_C_DECLARE_REDUCER macro expands to an
 *  anonymous `struct` declaration for a reducer object containing a view of
 *  type _value-type_, and the @ref CILK_C_INIT_REDUCER macro expands to a
 *  struct initializer.
 *
 *  @subsection reducers_c_reduce_func Reduce Functions
 *
 *  The reduce function for a reducer is called when a parallel execution
 *  strand terminates, to combine the values computed by the terminating
 *  strand and the strand to its left. It takes three arguments:
 *
 *  -   `void* reducer` - the address of the reducer.
 *  -   `void* left` - the address of the value for the left strand.
 *  -   `void* right` - the address of the value for the right (terminating)
 *                      strand.
 *
 *  It must apply the reducer's reduction operation to the `left` and `right`
 *  values, leaving the result in the `left` value. The `right` value is
 *  undefined after the reduce function call.
 *
 *  @subsection reducers_c_identity_func Identity Functions
 *
 *  The identity function for a reducer is called when a parallel execution
 *  strand begins, to initialize its value to the reducer's identity value. It
 *  takes two arguments:
 *
 *  -   `void* reducer` - the address of the reducer.
 *  -   `void* v` - the address of a freshly allocated block of memory of size
 *      `sizeof(value-type)`.
 *
 *  It must initialize the memory pointed to by `v` so that it contains the
 *  reducer's identity value.
 *
 *  @subsection reducers_c_destroy_func Destroy Functions
 *
 *  The destroy function for a reducer is called when a parallel execution
 *  strand terminates, to do any necessary cleanup before its value is
 *  deallocated. It takes two arguments:
 *
 *  -   `void* reducer` - the address of the reducer.
 *  -   `void* p` - the address of the value for the terminating strand.
 *
 *  It must release any resources belonging to the value pointed to by `p`, to
 *  avoid a resource leak when the memory containing the value is deallocated.
 *
 *  The runtime function `__cilkrts_hyperobject_noop_destroy` can be used for
 *  the destructor function if the reducer's values do not need any cleanup.
 *
 *  @subsection reducers_c_register Tell the Intel Cilk Plus Runtime About the
 *  Reducer
 *
 *  Call the @ref CILK_C_REGISTER_REDUCER macro to register the reducer with
 *  the Intel Cilk Plus runtime:
 *
 *      CILK_C_REGISTER_REDUCER(reducer-name);
 *
 *  The runtime will manage reducer values for all registered reducers when
 *  parallel execution strands begin and end.
 *
 *  @subsection reducers_c_update Update the Value Contained in the Reducer
 *
 *  The @ref REDUCER_VIEW macro returns a reference to the reducer's value for
 *  the current parallel strand:
 *
 *      REDUCER_VIEW(reducer-name) = REDUCER_VIEW(reducer-name) OP x;
 *
 *  C++ reducer views restrict access to the wrapped value so that it can only
 *  be modified in ways consistent with the reducer's operation. No such
 *  protection is provided for C reducers.  It is entirely the responsibility
 *  of the user to avoid modifying the value in any inappropriate way.
 *
 *  @subsection c_reducers_unregister Tell the Intel Cilk Plus Runtime That You Are
 *  Done with the Reducer
 *
 *  When the parallel computation is complete, call the @ref
 *  CILK_C_UNREGISTER_REDUCER macro to unregister the reducer with the
 *  Intel Cilk Plus runtime:
 *
 *      CILK_C_UNREGISTER_REDUCER(reducer-name);
 *
 *  The runtime will stop managing reducer values for the reducer.
 *
 *  @subsection c_reducers_retrieve Retrieve the Value from the Reducer
 *
 *  When the parallel computation is complete, use the @ref REDUCER_VIEW macro
 *  to retrieve the final value computed by the reducer.
 *
 *  @subsection reducers_c_example_custom Example - Creating and Using a
 *              Custom C Reducer
 *
 *  The `IntList` type represents a simple list of integers.
 *
 *      struct _intListNode {
 *          int value;
 *          _intListNode* next;
 *      } IntListNode;
 *      typedef struct { IntListNode* head; IntListNode* tail; } IntList;
 *
 *      // Initialize a list to be empty
 *      void IntList_init(IntList* list) { list->head = list->tail = 0; }
 *
 *      // Append an integer to the list
 *      void IntList_append(IntList* list, int x)
 *      {
 *          IntListNode* node = (IntListNode*) malloc(sizeof(IntListNode));
 *          if (list->tail) list->tail->next = node; else list->head = node;
 *          list->tail = node;
 *      }
 *
 *      // Append the right list to the left list, and leave the right list
 *      // empty
 *      void IntList_concat(IntList* left, IntList* right)
 *      {
 *          if (left->head) {
 *              left->tail->next = right->head;
 *              if (right->tail) left->tail = right->tail;
 *          }
 *          else {
 *              *left = *right;
 *          }
 *          IntList_init(*right);
 *      }
 *
 *  This code creates a reducer that supports creating an `IntList` by
 *  appending values to it.
 *
 *      void identity_IntList(void* reducer, void* list)
 *      {
 *          IntList_init((IntList*)list);
 *      }
 *
 *      void reduce_IntList(void* reducer, void* left, void* right)
 *      {
 *          IntList_concat((IntList*)left, (IntList*)right);
 *      }
 *
 *      CILK_C_DECLARE_REDUCER(IntList) my_list_int_reducer =
 *          CILK_C_INIT_REDUCER(IntList,
 *                              reduce_int_list,
 *                              identity_int_list,
 *                              __cilkrts_hyperobject_noop_destroy);
 *                              // Initial value omitted //
 *      ListInt_init(&REDUCER_VIEW(my_int_list_reducer));
 *
 *      CILK_C_REGISTER_REDUCER(my_int_list_reducer);
 *      cilk_for (int i = 0; i != n; ++i) {
 *          IntList_append(&REDUCER_VIEW(my_int_list_reducer), a[i]);
 *      }
 *      CILK_C_UNREGISTER_REDUCER(my_int_list_reducer);
 *
 *      IntList result = REDUCER_VIEW(my_int_list_reducer);
 *
 *  @section reducers_c_predefined Predefined C Reducers
 *
 *  Some of the predefined reducer classes in the Intel Cilk Plus library come with
 *  a set of predefined macros to provide the same capabilities in C.
 *  In general, two macros are provided for each predefined reducer family:
 *
 *  -   `CILK_C_REDUCER_operation(reducer-name, type-name, initial-value)` -
 *      Declares a reducer object named _reducer-name_ with initial value
 *      _initial-value_ to perform a reduction using the _operation_ on values
 *      of the type specified by _type-name_.  This is the equivalent of the
 *      general code described in @ref reducers_c_creation :
 *
 *          CILK_C_DECLARE_REDUCER(type) reducer-name =
 *              CILK_C_INIT_REDUCER(type, ..., initial-value);
 *
 *      where _type_ is the C type corresponding to _type_name_. See @ref
 *      reducers_c_type_names below for the _type-names_ that you can use.
 *
 *  -   `CILK_C_REDUCER_operation_TYPE(type-name)` - Expands to the `typedef`
 *      name for the type of the reducer object declared by
 *      `CILK_C_REDUCER_operation(reducer-name, type-name, initial-value)`.
 *
 *  See @ref reducers_c_example_predefined.
 *
 *  The predefined C reducers are:
 *
 *  |   Operation       |   Name        |   Documentation               |
 *  |-------------------|---------------|-------------------------------|
 *  |   addition        |   `OPADD`     |   @ref ReducersAdd            |
 *  |   bitwise AND     |   `OPAND`     |   @ref ReducersAnd            |
 *  |   bitwise OR      |   `OPOR`      |   @ref ReducersOr             |
 *  |   bitwise XOR     |   `OPXOR`     |   @ref ReducersXor            |
 *  |   multiplication  |   `OPMUL`     |   @ref ReducersMul            |
 *  |   minimum         |   `MIN`       |   @ref ReducersMinMax         |
 *  |   minimum & index |   `MIN_INDEX` |   @ref ReducersMinMax         |
 *  |   maximum         |   `MAX`       |   @ref ReducersMinMax         |
 *  |   maximum & index |   `MAX_INDEX` |   @ref ReducersMinMax         |
 *
 *  @subsection reducers_c_type_names Numeric Type Names
 *
 *  The type and function names created by the C reducer definition macros
 *  incorporate both the reducer kind (`opadd`, `opxor`, etc.) and the value
 *  type of the reducer (`int`, `double`, etc.). The value type is represented
 *  by a _numeric type name_ string. The types supported in C reducers, and
 *  their corresponding numeric type names, are given in the following table:
 *
 *  |   Type                |   Numeric Type Name           |
 *  |-----------------------|-------------------------------|
 *  |  `char`               |  `char`                       |
 *  |  `unsigned char`      |  `uchar`                      |
 *  |  `signed char`        |  `schar`                      |
 *  |  `wchar_t`            |  `wchar_t`                    |
 *  |  `short`              |  `short`                      |
 *  |  `unsigned short`     |  `ushort`                     |
 *  |  `int`                |  `int`                        |
 *  |  `unsigned int`       |  `uint`                       |
 *  |  `unsigned int`       |  `unsigned` (alternate name)  |
 *  |  `long`               |  `long`                       |
 *  |  `unsigned long`      |  `ulong`                      |
 *  |  `long long`          |  `longlong`                   |
 *  |  `unsigned long long` |  `ulonglong`                  |
 *  |  `float`              |  `float`                      |
 *  |  `double`             |  `double`                     |
 *  |  `long double`        |  `longdouble`                 |
 *
 *  @subsection reducers_c_example_predefined Example - Using a Predefined C
 *              Reducer
 *
 *  To compute the sum of all the values in an array of `unsigned int`:
 *
 *      CILK_C_REDUCER_OPADD(sum, uint, 0);
 *      CILK_C_REGISTER_REDUCER(sum);
 *      cilk_for(int i = 0; i != n; ++i) {
 *          REDUCER_VIEW(sum) += a[i];
 *      }
 *      CILK_C_UNREGISTER_REDUCER(sum);
 *      printf("The sum is %u\n", REDUCER_VIEW(sum));
 */


 /** @name C language reducer macros
 *
 *  These macros are used to declare and work with reducers in C code.
 *
 *  @see @ref page_reducers_in_c
 */
 //@{

/// @cond internal

/** @name Compound identifier macros.
 *
 *  These macros are used to construct an identifier by concatenating two or
 *  three identifiers.
 */
//@{

/** Expand to an identifier formed by concatenating two identifiers.
 */
#define __CILKRTS_MKIDENT(a,b) __CILKRTS_MKIDENT_IMP(a,b,)

/** Expand to an identifier formed by concatenating three identifiers.
 */
#define __CILKRTS_MKIDENT3(a,b,c) __CILKRTS_MKIDENT_IMP(a,b,c)

/** Helper macro to do the concatenation.
 */
#define __CILKRTS_MKIDENT_IMP(a,b,c) a ## b ## c

//@}

/** Compiler-specific keyword for the "type of" operator.
 */
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
# define _Typeof __typeof__
#endif

/** @name Predefined reducer function declaration macros.
 *
 *  These macros are used to create the function headers for the identity,
 *  reduction, and destructor functions for a builtin reducer family. The
 *  macro can be followed by a semicolon to create a declaration, or by a
 *  brace-enclosed body to create a definition.
 */
//@{

/** Create an identity function header.
 *
 *  @note The name of the function's value pointer parameter will always be `v`.
 *
 *  @param name The reducer family name.
 *  @param tn   The type name.
 */
#define __CILKRTS_DECLARE_REDUCER_IDENTITY(name,tn)  CILK_EXPORT         \
    void __CILKRTS_MKIDENT3(name,_identity_,tn)(void* key, void* v)

/** Create a reduction function header.
 *
 *  @param name The reducer family name.
 *  @param tn   The type name.
 *  @param l    The name to use for the function's left value pointer parameter.
 *  @param r    The name to use for the function's right value pointer 
 *              parameter.
 */
#define __CILKRTS_DECLARE_REDUCER_REDUCE(name,tn,l,r) CILK_EXPORT        \
    void __CILKRTS_MKIDENT3(name,_reduce_,tn)(void* key, void* l, void* r)

/** Create a destructor function header.
 *
 *  @param name The reducer family name.
 *  @param tn   The type name.
 *  @param p    The name to use for the function's value pointer parameter.
 */
#define __CILKRTS_DECLARE_REDUCER_DESTROY(name,tn,p) CILK_EXPORT         \
    void __CILKRTS_MKIDENT3(name,_destroy_,tn)(void* key, void* p)

//@}

/// @endcond


/***************************************************************************
 *              Real implementation
 ***************************************************************************/

/** Declaration of a C reducer structure type.
 *
 *  This macro expands into an anonymous structure declaration for a C reducer
 *  structure which contains a @a Type value. For example:
 *
 *      CILK_C_DECLARE_REDUCER(int) my_add_int_reducer =
 *          CILK_C_INIT_REDUCER(int, …);
 *
 *  @param Type The type of the value contained in the reducer object.
 *
 *  @see @ref reducers_c_creation
 */
#define CILK_C_DECLARE_REDUCER(Type) struct {                      \
        __cilkrts_hyperobject_base   __cilkrts_hyperbase;          \
        __CILKRTS_CACHE_ALIGN Type   value;                        \
    }

/** Initializer for a C reducer structure.
 *
 *  This macro expands into a brace-enclosed structure initializer for a C
 *  reducer structure that was declared with
 *  `CILK_C_DECLARE_REDUCER(Type)`. For example:
 *
 *      CILK_C_DECLARE_REDUCER(int) my_add_int_reducer =
 *          CILK_C_INIT_REDUCER(int,
 *                              add_int_reduce,
 *                              add_int_identity,
 *                              __cilkrts_hyperobject_noop_destroy,
 *                              0);
 *
 *  @param Type     The type of the value contained in the reducer object. Must
 *                  be the same as the @a Type argument of the
 *                  CILK_C_DECLARE_REDUCER macro call that created the
 *                  reducer.
 *  @param Reduce   The address of the @ref reducers_c_reduce_func
 *                  "reduce function" for the reducer.
 *  @param Identity The address of the @ref reducers_c_identity_func
 *                  "identity function" for the reducer.
 *  @param Destroy  The address of the @ref reducers_c_destroy_func
 *                  "destroy function" for the reducer.
 *  @param ...      The initial value for the reducer. (A single expression if
 *                  @a Type is a scalar type; a list of values if @a Type is a
 *                  struct or array type.)
 *
 *  @see @ref reducers_c_creation
 */

#define CILK_C_INIT_REDUCER(Type, Reduce, Identity, Destroy, ...)       \
    {   {   {   Reduce                                                  \
            ,   Identity                                                \
            ,   Destroy                                                 \
            ,   __cilkrts_hyperobject_alloc                             \
            ,   __cilkrts_hyperobject_dealloc                           \
            }                                                           \
        ,   0                                                           \
        ,   __CILKRTS_CACHE_LINE__                                      \
        ,   sizeof(Type)                                                \
        }                                                               \
    ,   __VA_ARGS__                                                     \
    }

/** Register a reducer with the Intel Cilk Plus runtime.
 *
 *  The runtime will manage reducer values for all registered reducers when
 *  parallel execution strands begin and end. For example:
 *
 *      CILK_C_REGISTER_REDUCER(my_add_int_reducer);
 *      cilk_for (int i = 0; i != n; ++i) {
 *          …
 *      }
 *
 *  @param Expr The reducer to be registered.
 *
 *  @see @ref page_reducers_in_c
 */
#define CILK_C_REGISTER_REDUCER(Expr) \
    __cilkrts_hyper_create(&(Expr).__cilkrts_hyperbase)

/** Unregister a reducer with the Intel Cilk Plus runtime.
 *
 *  The runtime will stop managing reducer values for a reducer after it is
 *  unregistered. For example:
 *
 *      cilk_for (int i = 0; i != n; ++i) {
 *          …
 *      }
 *      CILK_C_UNREGISTER_REDUCER(my_add_int_reducer);
 *
 *  @param Expr The reducer to be unregistered.
 *
 *  @see @ref page_reducers_in_c
 */
#define CILK_C_UNREGISTER_REDUCER(Expr) \
    __cilkrts_hyper_destroy(&(Expr).__cilkrts_hyperbase)

/** Get the current view for a reducer.
 *
 *  The `REDUCER_VIEW(reducer-name)` returns a reference to the reducer's
 *  value for the current parallel strand. This can be used to initialize the
 *  value of the reducer before it is used, to modify the value of the reducer
 *  on the current parallel strand, or to retrieve the final value of the
 *  reducer at the end of the parallel computation.
 *
 *      REDUCER_VIEW(my_add_int_reducer) = REDUCER_VIEW(my_add_int_reducer) + x;
 *
 *  @note C++ reducer views restrict access to the wrapped value so that it
 *  can only be modified in ways consistent with the reducer's operation. No
 *  such protection is provided for C reducers. It is entirely the
 *  responsibility of the user to refrain from modifying the value in any
 *  inappropriate way.
 *
 *  @param Expr The reducer whose value is to be returned.
 *
 *  @see @ref page_reducers_in_c
 */
#define REDUCER_VIEW(Expr) (*(_Typeof((Expr).value)*)               \
    __cilkrts_hyper_lookup(&(Expr).__cilkrts_hyperbase))

//@} C language reducer macros

#endif // CILK_REDUCER_H_INCLUDED
