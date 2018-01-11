/*  reducer_string.h                  -*- C++ -*-
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

/** @file reducer_string.h
 *
 *  @brief Defines classes for doing parallel string creation by appending.
 *
 *  @ingroup ReducersString
 *
 *  @see ReducersString
 */

#ifndef REDUCER_STRING_H_INCLUDED
#define REDUCER_STRING_H_INCLUDED

#include <cilk/reducer.h>
#include <string>
#include <list>

/** @defgroup ReducersString String Reducers
 *
 *  String reducers allow the creation of a string by concatenating a set of 
 *  strings or characters in parallel.
 *
 *  @ingroup Reducers
 *
 *  You should be familiar with @ref pagereducers "Cilk reducers", described in
 *  file reducers.md, and particularly with @ref reducers_using, before trying
 *  to use the information in this file.
 *
 *  @section redstring_usage Usage Example
 *
 *      vector<Data> data;
 *      void expensive_string_computation(const Data& x, string& s);
 *      cilk::reducer<cilk::op_string> r;
 *      cilk_for (int i = 0; i != data.size(); ++i) {
 *          string temp;
 *          expensive_string_computation(data[i], temp);
 *          *r += temp;
 *      }
 *      string result;
 *      r.move_out(result);
 *
 *  @section redstring_monoid The Monoid
 *
 *  @subsection redstring_monoid_values Value Set
 *
 *  The value set of a string reducer is the set of values of the class
 *  `std::basic_string<Char, Traits, Alloc>`, which we refer to as “the
 *  reducer’s string type”.
 *
 *  @subsection redstring_monoid_operator Operator
 *
 *  The operator of a string reducer is the string concatenation operator, 
 *  defined by the “`+`” binary operator on the reducer’s string type.
 *
 *  @subsection redstring_monoid_identity Identity
 *
 *  The identity value of a string reducer is the empty string, which is the 
 *  value of the expression
 *  `std::basic_string<Char, Traits, Alloc>([allocator])`.
 *
 *  @section redstring_operations Operations
 *
 *  In the operation descriptions below, the type name `String` refers to the
 *  reducer’s string type, `std::basic_string<Char, Traits, Alloc>`.
 *
 *  @subsection redstring_constructors Constructors
 *
 *  Any argument list which is valid for a `std::basic_string` constructor is
 *  valid for a string reducer constructor. The usual move-in constructor is
 *  also provided:
 *
 *      reducer(move_in(String& variable))
 *
 *  @subsection redstring_get_set Set and Get
 *
 *      r.set_value(const String& value)
 *      const String& = r.get_value() const
 *      r.move_in(String& variable)
 *      r.move_out(String& variable)
 *
 *  @subsection redstring_initial Initial Values
 *
 *  A string reducer with no constructor arguments, or with only an allocator
 *  argument, will initially contain the identity value, an empty string.
 *
 *  @subsection redstring_view_ops View Operations
 *
 *      *r += a
 *      r->append(a)
 *      r->append(a, b)
 *      r->push_back(a)
 *
 *  These operations on string reducer views are the same as the corresponding
 *  operations on strings.
 *
 *  @section redstring_performance Performance Considerations
 *
 *  String reducers work by creating a string for each view, collecting those
 *  strings in a list, and then concatenating them into a single result string
 *  at the end of the computation. This last step takes place in serial code,
 *  and necessarily takes time proportional to the length of the result string.
 *  Thus, a parallel string reducer cannot actually speed up the time spent
 *  directly creating the string. This trivial example would probably be slower
 *  (because of reducer overhead) than the corresponding serial code:
 *
 *      vector<string> a;
 *      reducer<op_string> r;
 *      cilk_for (int i = 0; i != a.length(); ++i) {
 *          *r += a[i];
 *      }
 *      string result;
 *      r.move_out(result);
 *
 *  What a string reducer _can_ do is to allow the _remainder_ of the
 *  computation to be done in parallel, without having to worry about managing
 *  the string computation.
 *
 *  The strings for new views are created (by the view identity constructor)
 *  using the same allocator as the string that was created when the reducer 
 *  was constructed. Note that this allocator is determined when the reducer is 
 *  constructed. The following two examples may have very different behavior:
 *
 *      string<Char, Traits, Allocator> a_string;
 *
 *      reducer< op_string<Char, Traits, Allocator> reducer1(move_in(a_string));
 *      ... parallel computation ...
 *      reducer1.move_out(a_string);
 *
 *      reducer< op_string<Char, Traits, Allocator> reducer2;
 *      reducer2.move_in(a_string);
 *      ... parallel computation ...
 *      reducer2.move_out(a_string);
 *
 *  *   `reducer1` will be constructed with the same allocator as `a_string`, 
 *      because the string was specified in the constructor. The `move_in`
 *      and `move_out` can therefore be done with a `swap` in constant time.
 *  *   `reducer2` will be constructed with a _default_ allocator of type
 *      `Allocator`, which may not be the same as the allocator of `a_string`.
 *      Therefore, the `move_in` and `move_out` may have to be done with a copy
 *      in _O(N)_ time.
 *
 *  (All instances of an allocator type with no internal state (like
 *  `std::allocator`) are “the same”. You only need to worry about the “same
 *  allocator” issue when you create string reducers with custom allocator
 *  types.)
 *
 *  @section redstring_types Type and Operator Requirements
 *
 *  `std::basic_string<Char, Traits, Alloc>` must be a valid type.
*/

namespace cilk {

/** @ingroup ReducersString */
//@{

/** The string append reducer view class.
 *
 *  This is the view class for reducers created with
 *  `cilk::reducer< cilk::op_basic_string<Type, Traits, Allocator> >`. It holds
 *  the accumulator variable for the reduction, and allows only append
 *  operations to be performed on it.
 *
 *  @note   The reducer “dereference” operation (`reducer::operator *()`) 
 *          yields a reference to the view. Thus, for example, the view class’s
 *          `append` operation would be used in an expression like
 *          `r->append(a)`, where `r` is a string append reducer variable.
 *
 *  @tparam Char        The string element type (not the string type).
 *  @tparam Traits      The character traits type.
 *  @tparam Alloc       The string allocator type.
 *
 *  @see ReducersString
 *  @see op_basic_string
 */
template<typename Char, typename Traits, typename Alloc>
class op_basic_string_view
{
    typedef std::basic_string<Char, Traits, Alloc>  string_type;
    typedef std::list<string_type>                  list_type;
    typedef typename string_type::size_type         size_type;

    // The view's value is represented by a list of strings and a single 
    // string. The value is the concatenation of the strings in the list with
    // the single string at the end. All string operations apply to the single
    // string; reduce operations cause lists of partial strings from multiple
    // strands to be combined.
    //
    mutable string_type                             m_string;
    mutable list_type                               m_list;

    // Before returning the value of the reducer, concatenate all the strings 
    // in the list with the single string.
    //
    void flatten() const
    {
        if (m_list.empty()) return;

        typename list_type::iterator i;

        size_type len = m_string.size();
        for (i = m_list.begin(); i != m_list.end(); ++i)
            len += i->size();

        string_type result(get_allocator());
        result.reserve(len);

        for (i = m_list.begin(); i != m_list.end(); ++i)
            result += *i;
        m_list.clear();

        result += m_string;
        result.swap(m_string);
    }

public:

    /** @name Monoid support.
     */
    //@{

    /// Required by @ref monoid_with_view
    typedef string_type value_type;

    /// Required by @ref op_string
    Alloc get_allocator() const
    {
        return m_string.get_allocator();
    }

    /** Reduction operation.
     *
     *  This function is invoked by the @ref op_basic_string monoid to combine
     *  the views of two strands when the right strand merges with the left 
     *  one. It appends the value contained in the right-strand view to the 
     *  value contained in the left-strand view, and leaves the value in the
     *  right-strand view undefined.
     *
     *  @param  right   A pointer to the right-strand view. (`this` points to
     *                  the left-strand view.)
     *
     *  @note   Used only by the @ref op_basic_string monoid to implement the
     *          monoid reduce operation.
     */
    void reduce(op_basic_string_view* right)
    {
        if (!right->m_string.empty() || !right->m_list.empty()) {
            // (list, string) + (right_list, right_string) =>
            //      (list + {string} + right_list, right_string)
            if (!m_string.empty()) {
                // simulate m_list.push_back(std::move(m_string))
                m_list.push_back(string_type(get_allocator()));
                m_list.back().swap(m_string);
            }
            m_list.splice(m_list.end(), right->m_list);
            m_string.swap(right->m_string);
        }
    }

    //@}

    /** @name Pass constructor arguments through to the string constructor.
     */
    //@{

    op_basic_string_view() : m_string() {}

    template <typename T1>
    op_basic_string_view(const T1& x1) : m_string(x1) {}

    template <typename T1, typename T2>
    op_basic_string_view(const T1& x1, const T2& x2) : m_string(x1, x2) {}

    template <typename T1, typename T2, typename T3>
    op_basic_string_view(const T1& x1, const T2& x2, const T3& x3) : m_string(x1, x2, x3) {}

    template <typename T1, typename T2, typename T3, typename T4>
    op_basic_string_view(const T1& x1, const T2& x2, const T3& x3, const T4& x4) :
        m_string(x1, x2, x3, x4) {}

    //@}

    /** Move-in constructor.
     */
    explicit op_basic_string_view(move_in_wrapper<value_type> w)
        : m_string(w.value().get_allocator())
    {
        m_string.swap(w.value());
    }

    /** @name @ref reducer support.
     */
    //@{

    void view_move_in(string_type& s)
    {
        m_list.clear();
        if (m_string.get_allocator() == s.get_allocator())
            // Equal allocators. Do a (fast) swap.
            m_string.swap(s);
        else
            // Unequal allocators. Do a (slow) copy.
            m_string = s;
        s.clear();
    }

    void view_move_out(string_type& s)
    {
        flatten();
        if (m_string.get_allocator() == s.get_allocator())
            // Equal allocators.  Do a (fast) swap.
            m_string.swap(s);
        else
            // Unequal allocators.  Do a (slow) copy.
            s = m_string;
        m_string.clear();
    }

    void view_set_value(const string_type& s) 
        { m_list.clear(); m_string = s; }

    string_type const& view_get_value()     const 
        { flatten(); return m_string; }

    string_type      & view_get_reference()       
        { flatten(); return m_string; }

    string_type const& view_get_reference() const 
        { flatten(); return m_string; }

    //@}

    /** @name View modifier operations.
     *
     *  @details These simply wrap the corresponding operations on the underlying string.
     */
    //@{

    template <typename T>
    op_basic_string_view& operator +=(const T& x)
        { m_string += x; return *this; }

    template <typename T1>
    op_basic_string_view& append(const T1& x1)
        { m_string.append(x1); return *this; }

    template <typename T1, typename T2>
    op_basic_string_view& append(const T1& x1, const T2& x2)
        { m_string.append(x1, x2); return *this; }

    template <typename T1, typename T2, typename T3>
    op_basic_string_view& append(const T1& x1, const T2& x2, const T3& x3)
        { m_string.append(x1, x2, x3); return *this; }

    void push_back(const Char x) { m_string.push_back(x); }

    //@}
};


/** String append monoid class. Instantiate the cilk::reducer template class
 *  with an op_basic_string monoid to create a string append reducer class. For
 *  example, to concatenate a collection of standard strings:
 *
 *      cilk::reducer< cilk::op_basic_string<char> > r;
 *
 *  @tparam Char    The string element type (not the string type).
 *  @tparam Traits  The character traits type.
 *  @tparam Alloc   The string allocator type.
 *  @tparam Align   If `false` (the default), reducers instantiated on this
 *                  monoid will be naturally aligned (the Cilk library 1.0
 *                  behavior). If `true`, reducers instantiated on this monoid
 *                  will be cache-aligned for binary compatibility with 
 *                  reducers in Cilk library version 0.9.
 *
 *  @see ReducersString
 *  @see op_basic_string_view
 *  @see reducer_basic_string
 *  @see op_string
 *  @see op_wstring
 */
template<typename Char,
         typename Traits = std::char_traits<Char>,
         typename Alloc = std::allocator<Char>,
         bool     Align = false>
class op_basic_string : 
    public monoid_with_view< op_basic_string_view<Char, Traits, Alloc>, Align >
{
    typedef monoid_with_view< op_basic_string_view<Char, Traits, Alloc>, Align >
            base;
    Alloc m_allocator;

public:

    /** View type of the monoid.
     */
    typedef typename base::view_type view_type;

    /** Constructor.
     *
     *  There is no default constructor for string monoids, because the
     *  allocator must always be specified.
     *
     *  @param  allocator   The list allocator to be used when
     *                      identity-constructing new views.
     */
    op_basic_string(const Alloc& allocator = Alloc()) : m_allocator(allocator)
    {}

    /** Create an identity view.
     *
     *  String view identity constructors take the string allocator as an
     *  argument.
     *
     *  @param v    The address of the uninitialized memory in which the view
     *              will be constructed.
     */
    void identity(view_type *v) const { ::new((void*) v) view_type(m_allocator); }

    /** @name Construct functions
     *
     *  A string append reduction monoid must have a copy of the allocator of
     *  the leftmost view’s string, so that it can use it in the `identity`
     *  operation. This, in turn, requires that string reduction monoids have a
     *  specialized `construct()` function.
     *
     *  All string reducer monoid `construct()` functions first construct the
     *  leftmost view, using the arguments that were passed in from the reducer
     *  constructor. They then call the view’s `get_allocator()` function to
     *  get the string allocator from the string in the leftmost view, and pass
     *  that to the monoid constructor.
     */
    //@{

    static void construct(op_basic_string* monoid, view_type* view)
        { provisional( new ((void*)view) view_type() ).confirm_if(
            new ((void*)monoid) op_basic_string(view->get_allocator()) ); }

    template <typename T1>
    static void construct(op_basic_string* monoid, view_type* view, const T1& x1)
        { provisional( new ((void*)view) view_type(x1) ).confirm_if(
            new ((void*)monoid) op_basic_string(view->get_allocator()) ); }

    template <typename T1, typename T2>
    static void construct(op_basic_string* monoid, view_type* view, const T1& x1, const T2& x2)
        { provisional( new ((void*)view) view_type(x1, x2) ).confirm_if(
            new ((void*)monoid) op_basic_string(view->get_allocator()) ); }

    template <typename T1, typename T2, typename T3>
    static void construct(op_basic_string* monoid, view_type* view, const T1& x1, const T2& x2,
                            const T3& x3)
        { provisional( new ((void*)view) view_type(x1, x2, x3) ).confirm_if(
            new ((void*)monoid) op_basic_string(view->get_allocator()) ); }

    template <typename T1, typename T2, typename T3, typename T4>
    static void construct(op_basic_string* monoid, view_type* view, const T1& x1, const T2& x2,
                            const T3& x3, const T4& x4)
        { provisional( new ((void*)view) view_type(x1, x2, x3, x4) ).confirm_if(
            new ((void*)monoid) op_basic_string(view->get_allocator()) ); }

    //@}
};


/** Convenience typedef for 8-bit strings
 */
typedef op_basic_string<char> op_string;
    
/** Convenience typedef for 16-bit strings
 */
typedef op_basic_string<wchar_t> op_wstring;


/** Deprecated string append reducer class.
 *
 *  reducer_basic_string is the same as @ref reducer<@ref op_basic_string>,
 *  except that reducer_basic_string is a proxy for the contained view, so that
 *  accumulator variable update operations can be applied directly to the
 *  reducer. For example, a value is appended to a `reducer<%op_basic_string>`
 *  with `r->push_back(a)`, but a value can be appended to  a `%reducer_opand`
 *  with `r.push_back(a)`.
 *
 *  @deprecated Users are strongly encouraged to use `reducer<monoid>`
 *              reducers rather than the old wrappers like reducer_basic_string. 
 *              The `reducer<monoid>` reducers show the reducer/monoid/view
 *              architecture more clearly, are more consistent in their
 *              implementation, and present a simpler model for new
 *              user-implemented reducers.
 *
 *  @note   Implicit conversions are provided between `%reducer_basic_string` 
 *          and `reducer<%op_basic_string>`. This allows incremental code
 *          conversion: old code that used `%reducer_basic_string` can pass a
 *          `%reducer_basic_string` to a converted function that now expects a
 *          pointer or reference to a `reducer<%op_basic_string>`, and vice
 *          versa.
 *
 *  @tparam Char        The string element type (not the string type).
 *  @tparam Traits      The character traits type.
 *  @tparam Alloc       The string allocator type.
 *
 *  @see op_basic_string
 *  @see reducer
 *  @see ReducersString
 */
template<typename Char,
         typename Traits = std::char_traits<Char>,
         typename Alloc = std::allocator<Char> >
class reducer_basic_string : 
    public reducer< op_basic_string<Char, Traits, Alloc, true> >
{
    typedef reducer< op_basic_string<Char, Traits, Alloc, true> > base;
    using base::view;
public:

    /// The reducer’s string type.
    typedef typename base::value_type       string_type;

    /// The reducer’s primitive component type.
    typedef Char                            basic_value_type;

    /// The string size type.
    typedef typename string_type::size_type size_type;

    /// The view type for the reducer.
    typedef typename base::view_type        View;

    /// The monoid type for the reducer.
    typedef typename base::monoid_type      Monoid;


    /** @name Constructors
     */
    //@{
    
    /** @name Forward constructor calls to the base class.
     *
     *  All basic_string constructor forms are supported.
     */
    //@{
    reducer_basic_string() {}

    template <typename T1>
    reducer_basic_string(const T1& x1) : 
        base(x1) {}

    template <typename T1, typename T2>
    reducer_basic_string(const T1& x1, const T2& x2) : 
        base(x1, x2) {}

    template <typename T1, typename T2, typename T3>
    reducer_basic_string(const T1& x1, const T2& x2, const T3& x3) : 
        base(x1, x2, x3) {}

    template <typename T1, typename T2, typename T3, typename T4>
    reducer_basic_string(const T1& x1, const T2& x2, const T3& x3, const T4& x4) :
        base(x1, x2, x3, x4) {}
    //@}

    /** Allow mutable access to the string within the current view.
     *
     *  @warning    If this method is called before the parallel calculation is 
     *              complete, the string returned by this method will be a
     *              partial result.
     *
     *  @returns    A mutable reference to the string within the current view.
     */
    string_type &get_reference() 
        { return view().view_get_reference(); }

    /** Allow read-only access to the string within the current view.
     *
     *  @warning    If this method is called before the parallel calculation is
     *              complete, the string returned by this method will be a
     *              partial result.
     *
     *  @returns    A const reference to the string within the current view.
     */
    string_type const &get_reference() const 
        { return view().view_get_reference(); }

    /** @name Append to the string.
     *
     *  These operations are simply forwarded to the view.
     */
    //@{
    void append(const Char *ptr)
        { view().append(ptr); }
    void append(const Char *ptr, size_type count)
        { view().append(ptr, count); }
    void append(const string_type &str, size_type offset, size_type count)
        { view().append(str, offset, count); }
    void append(const string_type &str)
        { view().append(str); }
    void append(size_type count, Char ch)
        { view().append(count, ch); }

    // Append to the string
    reducer_basic_string<Char, Traits, Alloc> &operator+=(Char ch)
        { view() += ch; return *this; }
    reducer_basic_string<Char, Traits, Alloc> &operator+=(const Char *ptr)
        { view() += ptr; return *this; }
    reducer_basic_string<Char, Traits, Alloc> &operator+=(const string_type &right)
        { view() += right; return *this; }
    //@}

    /** @name Dereference
     *  @details Dereferencing a wrapper is a no-op. It simply returns the
     *  wrapper. Combined with the rule that the wrapper forwards view
     *  operations to its contained view, this means that view operations can
     *  be written the same way on reducers and wrappers, which is convenient
     *  for incrementally converting old code using wrappers to use reducers
     *  instead. That is:
     *
     *      reducer<op_string> r;
     *      r->push_back(a);    // r-> returns the view
     *                          // push_back() is a view member function
     *
     *      reducer_string w;
     *      w->push_back(a);    // *w returns the wrapper
     *                          // push_back() is a wrapper member function
     *                          // that calls the corresponding view function
     */
    //@{
    reducer_basic_string&       operator*()       { return *this; }
    reducer_basic_string const& operator*() const { return *this; }

    reducer_basic_string*       operator->()       { return this; }
    reducer_basic_string const* operator->() const { return this; }
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
    operator reducer< op_basic_string<Char, Traits, Alloc, false> >& ()
    {
        return *reinterpret_cast< reducer< 
            op_basic_string<Char, Traits, Alloc, false> >* 
        >(this);
    }
    operator const reducer< op_basic_string<Char, Traits, Alloc, false> >& () const
    {
        return *reinterpret_cast< const reducer<
            op_basic_string<Char, Traits, Alloc, false> >* 
        >(this);
    }
    //@}
};


/** Convenience typedef for 8-bit strings
 */
typedef reducer_basic_string<char> reducer_string;

/** Convenience typedef for 16-bit strings
 */
typedef reducer_basic_string<wchar_t> reducer_wstring;

/// @cond internal

/// @cond internal
/** Metafunction specialization for reducer conversion.
 *
 *  This specialization of the @ref legacy_reducer_downcast template class 
 *  defined in reducer.h causes the `reducer< op_basic_string<Char> >` class to
 *  have an `operator reducer_basic_string<Char>& ()` conversion operator that
 *  statically downcasts the `reducer<op_basic_string>` to the corresponding
 *  `reducer_basic_string` type. (The reverse conversion, from 
 *  `reducer_basic_string` to `reducer<op_basic_string>`, is just an upcast,
 *  which is provided for free by the language.)
 *
 *  @ingroup ReducersString
 */
template<typename Char, typename Traits, typename Alloc, bool Align>
struct legacy_reducer_downcast<
    reducer<op_basic_string<Char, Traits, Alloc, Align> > >
{
    typedef reducer_basic_string<Char, Traits, Alloc> type;
};

/// @endcond

//@}

} // namespace cilk

#endif //  REDUCER_STRING_H_INCLUDED
