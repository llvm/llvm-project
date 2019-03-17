/*  reducer_vector.h                  -*- C++ -*-
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

/** @file reducer_vector.h
 *
 *  @brief Defines classes for doing parallel vector creation by appending.
 *
 *  @ingroup ReducersVector
 *
 *  @see ReducersVector
 */

#ifndef REDUCER_VECTOR_H_INCLUDED
#define REDUCER_VECTOR_H_INCLUDED

#include <cilk/reducer.h>
#include <vector>
#include <list>

/** @defgroup ReducersVector Vector Reducers
 *
 *  Vector reducers allow the creation of a standard vector by
 *  appending a set of elements in parallel.
 *
 *  @ingroup Reducers
 *
 *  You should be familiar with @ref pagereducers "Intel(R) Cilk(TM) Plus reducers",
 *  described in file `reducers.md`, and particularly with @ref reducers_using,
 *  before trying to use the information in this file.
 *
 *  @section redvector_usage Usage Example
 *
 *      typedef ... SourceData;
 *      typedef ... ResultData;
 *      vector<SourceData> input;
 *      ResultData expensive_computation(const SourceData& x);
 *      cilk::reducer< cilk::op_vector<ResultData> > r;
 *      cilk_for (int i = 0; i != input.size(); ++i) {
 *          r->push_back(expensive_computation(input[i]));
 *      }
 *      vector result;
 *      r.move_out(result);
 *
 *  @section redvector_monoid The Monoid
 *
 *  @subsection redvector_monoid_values Value Set
 *
 *  The value set of a vector reducer is the set of values of the class
 *  `std::vector<Type, Alloc>`, which we refer to as "the reducer's vector
 *  type".
 *
 *  @subsection redvector_monoid_operator Operator
 *
 *  The operator of a vector reducer is vector concatenation.
 *
 *  @subsection redvector_monoid_identity Identity
 *
 *  The identity value of a vector reducer is the empty vector, which is the
 *  value of the expression `std::vector<Type, Alloc>([allocator])`.
 *
 *  @section redvector_operations Operations
 *
 *  In the operation descriptions below, the type name `Vector` refers to
 *  the reducer's vector type, `std::vector<Type, Alloc>`.
 *
 *  @subsection redvector_constructors Constructors
 *
 *  Any argument list which is valid for a `std::vector` constructor is valid
 *  for a vector reducer constructor. The usual move-in constructor is also
 *  provided:
 *
 *      reducer(move_in(Vector& variable))
 *
 *  @subsection redvector_get_set Set and Get
 *
 *      void r.set_value(const Vector& value)
 *      const Vector& = r.get_value() const
 *      void r.move_in(Vector& variable)
 *      void r.move_out(Vector& variable)
 *
 *  @subsection redvector_initial Initial Values
 *
 *  A vector reducer with no constructor arguments, or with only an allocator
 *  argument, will initially contain the identity value, an empty vector.
 *
 *  @subsection redvector_view_ops View Operations
 *
 *  The view of a vector reducer provides the following member functions:
 *
 *      void push_back(const Type& element)
 *      void insert_back(const Type& element)
 *      void insert_back(Vector::size_type n, const Type& element)
 *      template <typename Iter> void insert_back(Iter first, Iter last)
 *
 *  The `push_back` functions is the same as the corresponding `std::vector`
 *  function. The `insert_back` function is the same as the `std::vector`
 *  `insert` function, with the first parameter fixed to the end of the vector.
 *
 *  @section redvector_performance Performance Considerations
 *
 *  Vector reducers work by creating a vector for each view, collecting those
 *  vectors in a list, and then concatenating them into a single result vector
 *  at the end of the computation. This last step takes place in serial code,
 *  and necessarily takes time proportional to the length of the result vector.
 *  Thus, a parallel vector reducer cannot actually speed up the time spent
 *  directly creating the vector. This trivial example would probably be slower
 *  (because of reducer overhead) than the corresponding serial code:
 *
 *      vector<T> a;
 *      reducer<op_vector<T> > r;
 *      cilk_for (int i = 0; i != a.length(); ++i) {
 *          r->push_back(a[i]);
 *      }
 *      vector<T> result;
 *      r.move_out(result);
 *
 *  What a vector reducer _can_ do is to allow the _remainder_ of the
 *  computation to be done in parallel, without having to worry about
 *  managing the vector computation.
 *
 *  The vectors for new views are created (by the view identity constructor)
 *  using the same allocator as the vector that was created when the reducer
 *  was constructed. Note that this allocator is determined when the reducer
 *  is constructed. The following two examples may have very different
 *  behavior:
 *
 *      vector<Type, Allocator> a_vector;
 *
 *      reducer< op_vector<Type, Allocator> reducer1(move_in(a_vector));
 *      ... parallel computation ...
 *      reducer1.move_out(a_vector);
 *
 *      reducer< op_vector<Type, Allocator> reducer2;
 *      reducer2.move_in(a_vector);
 *      ... parallel computation ...
 *      reducer2.move_out(a_vector);
 *
 *  *   `reducer1` will be constructed with the same allocator as `a_vector`,
 *      because the vector was specified in the constructor. The `move_in`
 *      and`move_out` can therefore be done with a `swap` in constant time.
 *  *   `reducer2` will be constructed with a _default_ allocator of type
 *      `Allocator`, which may not be the same as the allocator of `a_vector`.
 *      Therefore, the `move_in` and `move_out` may have to be done with a
 *      copy in _O(N)_ time.
 *
 *  (All instances of an allocator class with no internal state (like
 *  `std::allocator`) are "the same". You only need to worry about the "same
 *  allocator" issue when you create vector reducers with a custom allocator
 *  class that has data members.)
 *
 *  @section redvector_types Type and Operator Requirements
 *
 *  `std::vector<Type, Alloc>` must be a valid type.
*/

namespace cilk {

/** @ingroup ReducersVector */
//@{

/** @brief The vector reducer view class.
 *
 *  This is the view class for reducers created with
 *  `cilk::reducer< cilk::op_vector<Type, Allocator> >`. It holds the
 *  accumulator variable for the reduction, and allows only append operations
 *  to be performed on it.
 *
 *  @note   The reducer "dereference" operation (`reducer::operator *()`)
 *          yields a reference to the view. Thus, for example, the view
 *          class's `push_back` operation would be used in an expression like
 *          `r->push_back(a)`, where `r` is a vector reducer variable.
 *
 *  @tparam Type        The vector element type (not the vector type).
 *  @tparam Alloc       The vector allocator type.
 *
 *  @see @ref ReducersVector
 *  @see op_vector
 */
template<typename Type, typename Alloc>
class op_vector_view
{
    typedef std::vector<Type, Alloc>                vector_type;
    typedef std::list<vector_type, typename Alloc::template rebind<vector_type>::other>
                                                    list_type;
    typedef typename vector_type::size_type         size_type;

    // The view's value is represented by a list of vectors and a single
    // vector. The value is the concatenation of the vectors in the list with
    // the single vector at the end. All vector operations apply to the single
    // vector; reduce operations cause lists of partial vectors from multiple
    // strands to be combined.
    //
    mutable vector_type                             m_vector;
    mutable list_type                               m_list;

    // Before returning the value of the reducer, concatenate all the vectors
    // in the list with the single vector.
    //
    void flatten() const
    {
        if (m_list.empty()) return;

        typename list_type::iterator i;

        size_type len = m_vector.size();
        for (i = m_list.begin(); i != m_list.end(); ++i)
            len += i->size();

        vector_type result(get_allocator());
        result.reserve(len);

        for (i = m_list.begin(); i != m_list.end(); ++i)
            result.insert(result.end(), i->begin(), i->end());
        m_list.clear();

        result.insert(result.end(), m_vector.begin(), m_vector.end());
        result.swap(m_vector);
    }

public:

    /** @name Monoid support.
     */
    //@{

    /// Required by cilk::monoid_with_view
    typedef vector_type value_type;

    /// Required by @ref op_vector
    Alloc get_allocator() const
    {
        return m_vector.get_allocator();
    }

    /** Reduces the views of two strands.
     *
     *  This function is invoked by the @ref op_vector monoid to combine
     *  the views of two strands when the right strand merges with the left
     *  one. It appends the value contained in the right-strand view to the
     *  value contained in the left-strand view, and leaves the value in the
     *  right-strand view undefined.
     *
     *  @param  other   A pointer to the right-strand view. (`this` points to
     *                  the left-strand view.)
     *
     *  @note   Used only by the @ref op_vector monoid to implement the
     *          monoid reduce operation.
     */
    void reduce(op_vector_view* other)
    {
        if (!other->m_vector.empty() || !other->m_list.empty()) {
            // (list, string) + (other_list, other_string) =>
            //      (list + {string} + other_list, other_string)
            if (!m_vector.empty()) {
                // simulate m_list.push_back(std::move(m_vector))
                m_list.push_back(vector_type(get_allocator()));
                m_list.back().swap(m_vector);
            }
            m_list.splice(m_list.end(), other->m_list);
            m_vector.swap(other->m_vector);
        }
    }

    //@}

    /** @name Passes constructor arguments to the vector constructor.
     */
    //@{

    op_vector_view() :
        m_vector(), m_list(get_allocator()) {}

    template <typename T1>
    op_vector_view(const T1& x1) :
        m_vector(x1), m_list(get_allocator()) {}

    template <typename T1, typename T2>
    op_vector_view(const T1& x1, const T2& x2) :
        m_vector(x1, x2), m_list(get_allocator()) {}

    template <typename T1, typename T2, typename T3>
    op_vector_view(const T1& x1, const T2& x2, const T3& x3) :
        m_vector(x1, x2, x3), m_list(get_allocator()) {}

    template <typename T1, typename T2, typename T3, typename T4>
    op_vector_view(const T1& x1, const T2& x2, const T3& x3, const T4& x4) :
        m_vector(x1, x2, x3, x4), m_list(get_allocator()) {}

    //@}

    /** Move-in constructor.
     */
    explicit op_vector_view(cilk::move_in_wrapper<value_type> w) :
        m_vector(w.value().get_allocator()),
        m_list(w.value().get_allocator())
    {
        m_vector.swap(w.value());
    }

    /** @name Reducer support.
     */
    //@{

    void view_move_in(vector_type& v)
    {
        m_list.clear();
        if (get_allocator() == v.get_allocator()) {
            // Equal allocators. Do a (fast) swap.
            m_vector.swap(v);
        }
        else {
            // Unequal allocators. Do a (slow) copy.
            m_vector = v;
        }
        v.clear();
    }

    void view_move_out(vector_type& v)
    {
        flatten();
        if (get_allocator() == v.get_allocator()) {
            // Equal allocators.  Do a (fast) swap.
            m_vector.swap(v);
        }
        else {
            // Unequal allocators.  Do a (slow) copy.
            v = m_vector;
        m_vector.clear();
        }
    }

    void view_set_value(const vector_type& v)
    {
        m_list.clear();
        m_vector = v;
    }

    vector_type const& view_get_value()     const
    {
        flatten();
        return m_vector;
    }

    typedef vector_type const& return_type_for_get_value;

    //@}

    /** @name View modifier operations.
     *
     *  @details These simply wrap the corresponding operations on the
     *  underlying vector.
     */
    //@{

    /** Adds an element at the end of the list.
     *
     *  Equivalent to `vector.push_back(…)`
     */
    void push_back(const Type x)
    {
        m_vector.push_back(x);
    }

    /** @name Insert elements at the end of the vector.
     *
     *  Equivalent to `vector.insert(vector.end(), …)`
     */
    //@{

    void insert_back(const Type& element)
        { m_vector.insert(m_vector.end(), element); }

    void insert_back(typename vector_type::size_type n, const Type& element)
        { m_vector.insert(m_vector.end(), n, element); }

    template <typename Iter>
    void insert_back(Iter first, Iter last)
        { m_vector.insert(m_vector.end(), first, last); }

    //@}

    //@}
};


/** @brief The vector append monoid class.
 *
 *  Instantiate the cilk::reducer template class with an op_vector monoid to
 *  create a vector reducer class. For example, to concatenate a
 *  collection of integers:
 *
 *      cilk::reducer< cilk::op_vector<int> > r;
 *
 *  @tparam Type        The vector element type (not the vector type).
 *  @tparam Alloc       The vector allocator type.
 *
 *  @see ReducersVector
 *  @see op_vector_view
 *  @ingroup ReducersVector
 */
template<typename Type, typename Alloc = std::allocator<Type> >
class op_vector :
    public cilk::monoid_with_view< op_vector_view<Type, Alloc>, false >
{
    typedef cilk::monoid_with_view< op_vector_view<Type, Alloc>, false > base;
    typedef provisional_guard<typename base::view_type> view_guard;

    // The allocator to be used when constructing new views.
    Alloc m_allocator;

public:

    /// View type.
    typedef typename base::view_type view_type;

    /** Constructor.
     *
     *  There is no default constructor for vector monoids, because the
     *  allocator must always be specified.
     *
     *  @param  allocator   The list allocator to be used when
     *                      identity-constructing new views.
     */
    op_vector(const Alloc& allocator = Alloc()) : m_allocator(allocator) {}

    /** Creates an identity view.
     *
     *  Vector view identity constructors take the vector allocator as an
     *  argument.
     *
     *  @param v    The address of the uninitialized memory in which the view
     *              will be constructed.
     */
    void identity(view_type *v) const
    {
        ::new((void*) v) view_type(m_allocator);
    }

    /** @name construct functions
     *
     *  A vector append monoid must have a copy of the allocator of
     *  the leftmost view's vector, so that it can use it in the `identity`
     *  operation. This, in turn, requires that vector append monoids have a
     *  specialized `construct()` function.
     *
     *  All vector append monoid `construct()` functions first construct the
     *  leftmost view, using the arguments that were passed in from the reducer
     *  constructor. They then call the view's `get_allocator()` function to
     *  get the vector allocator from the vector in the leftmost view, and pass
     *  that to the monoid constructor.
     */
    //@{

    static void construct(op_vector* monoid, view_type* view)
    {
        view_guard vg( new((void*) view) view_type() );
        vg.confirm_if( new((void*) monoid) op_vector(view->get_allocator()) ); 
    }

    template <typename T1>
    static void construct(op_vector* monoid, view_type* view, const T1& x1)
    {
        view_guard vg( new((void*) view) view_type(x1) );
        vg.confirm_if( new((void*) monoid) op_vector(view->get_allocator()) ); 
    }

    template <typename T1, typename T2>
    static void construct(op_vector* monoid, view_type* view,
        const T1& x1, const T2& x2)
    {
        view_guard vg( new((void*) view) view_type(x1, x2) );
        vg.confirm_if( new((void*) monoid) op_vector(view->get_allocator()) ); 
    }

    template <typename T1, typename T2, typename T3>
    static void construct(op_vector* monoid, view_type* view,
        const T1& x1, const T2& x2, const T3& x3)
    {
        view_guard vg( new((void*) view) view_type(x1, x2, x3) );
        vg.confirm_if( new((void*) monoid) op_vector(view->get_allocator()) ); 
    }

    //@}
};


} // namespace cilk

#endif //  REDUCER_VECTOR_H_INCLUDED
