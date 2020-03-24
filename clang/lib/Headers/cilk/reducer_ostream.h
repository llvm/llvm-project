/*  reducer_ostream.h                  -*- C++ -*-
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

/** @file reducer_ostream.h
 *
 *  @brief Defines a class for writing to an ostream in parallel.
 *
 *  @ingroup ReducersOstream
 *
 *  @see @ref ReducersOstream
 */

#ifndef REDUCER_OSTREAM_H_INCLUDED
#define REDUCER_OSTREAM_H_INCLUDED

#include <cilk/reducer.h>
#include <ostream>
#include <sstream>

/** @defgroup ReducersOstream Ostream Reducers
 *
 *  Ostream reducers allow multiple strands to write to an ostream in parallel.
 *
 *  @ingroup Reducers
 *
 *  You should be familiar with @ref pagereducers "Intel(R) Cilk(TM) Plus reducers",
 *  described in file reducers.md, and particularly with @ref reducers_using,
 *  before trying to use the information in this file.
 *
 *  @section redostream_usage Usage Example
 *
 *  One of the most common debugging techniques is adding `print` statements
 *  to the code being debugged. When the code is parallelized, the results can
 *  be less than satisfactory, as output from multiple strands is mingled in an
 *  unpredictable way. Like other reducers, an ostream reducer requires minimal
 *  recoding to guarantee that the output from parallelized computation will be
 *  ordered the same as though the computation were executed serially.
 *
 *      cilk::reducer<cilk::op_ostream> r(std::cerr);
 *      cilk_for (int i = 0; i != data.size(); ++i) {
 *          *r << "Iteration " << i << ":\n";
 *          ... some computation ...
 *          *r << "   Step 1:" << some information;
 *          ... some more computation ...
 *          *r << "   Step 2:" << some more information;
 *          ... still more computation ...
 *          *r << "   Step 3:" << still more information;
 *      }
 *
 *  Output on standard error:
 *
 *      Iteration 1:
 *          Step 1: ...
 *          Step 2: ...
 *          Step 3: ...
 *      Iteration 2:
 *          Step 1: ...
 *          Step 2: ...
 *          Step 3: ...
 *      Iteration 3:
 *          Step 1: ...
 *          Step 2: ...
 *          Step 3: ...
 *      ...
 *
 *  @section redostream_overview Overview
 *
 *  An "ostream reducer" is not really a reducer. It uses the reducer
 *  technology to coordinate operations on parallel strands to achieve
 *  the same behavior in a parallel computation that would be seen in a
 *  serial computation, but it does not have a monoid. It has a "monoid
 *  class," because that is part of the implementation framework, but it
 *  does not represent a mathematical monoid: there is no value type, no
 *  associative operation, and no identity value. The reducer is used for
 *  its side effect rather than to construct a value.
 *
 *  You might think of an ostream reducer as a relative of a
 *  @ref ReducersString "string reducer" which uses stream output
 *  syntax (`stream << value`) instead of string append syntax
 *  (`string += value`), and which writes its result string to an
 *  ostream instead of making it available as the reducer value.
 *
 *  Another difference is that "real" reducers protect their contained
 *  value quite strongly from improper access by the user. Ostream reducers,
 *  on the other hand, pretty much have to expose the ostream, since normal
 *  use of an ostream involves accessing its internal state. Furthermore,
 *  the ostream reducer just coordinates output to an existing ostream -
 *  there is nothing to keep the user from writing directly to the attached
 *  stream, with unpredictable results.
 *
 *  @section redostream_operations Operations
 *
 *  In the operation descriptions below, the type name `Ostream` refers to the
 *  reducer's ostream type, `std::basic_ostream<Char, Traits>`.
 *
 *  @subsection redostream_constructors Constructors
 *
 *  The only constructor is
 *
 *      reducer(const Ostream& os)
 *
 *  This creates a reducer that is associated with the existing ostream `os`.
 *  Anything "written to" the reducer will (eventually) be written to `os`.
 *
 *  @subsection redostream_get_set Set and Get
 *
 *  Just as a stream does not have a "value," neither does an ostream
 *  reducer. Therefore, none of the usual `set_value`, `get_value`,
 *  `move_in`, or `move_out` functions are available for ostream reducers.
 *
 *  @subsection redostream_initial Initial Values
 *
 *  Ostream reducers do not have default constructors.
 *
 *  @subsection redostream_view_ops View Operations
 *
 *  An ostream reducer view is actually a kind of `std::ostream`. Therefore,
 *  any operation that can be used on an ostream can be used on an ostream
 *  reducer view. For example:
 *
 *      reducer<op_ostream> r(cout);
 *      *r << setw(5) << (x=1) << endl;
 *
 *
 *  @section redostream_performance Performance Considerations
 *
 *  Ostream reducers work by creating a string stream for each non-leftmost
 *  view. When two strands are merged, the contents of the string buffer of the
 *  right view are written to the left view. Since all non-leftmost strands are
 *  eventually merged, all output is eventually written to the associated
 *  ostream.
 *
 *  This implementation has two consequences.
 *
 *  First, all output written to an ostream reducer on a stolen strand is kept
 *  in memory (in a string buffer) until the strand is merged with the leftmost
 *  strand. This means that some portion of the output written to an ostream
 *  reducer during a parallel computation - half of the total output, on
 *  average - will temporarily be held in memory during the computation.
 *  Obviously, ostream reducers will work better for small and moderate amounts
 *  of output.
 *
 *  Second, buffered ostream reducer content must be copied at every merge.
 *  The total amount of copying is potentially proportional to the total amount
 *  of output multiplied by the number of strands stolen during the computation.
 *
 *  In short, writing to an ostream in a parallel computation with an ostream
 *  reducer will always be less efficient than writing the same output directly
 *  to the ostream in a serial computation. The value of the ostream
 *  reducer is not in the writing of the ostream itself, but in removing the
 *  race and serialization obstacles that the ostream output would cause in an
 *  otherwise parallelizable computation.
 *
 *
 *  @section redostream_state Stream State
 *
 *  The reducer implementation can correctly order the output that is written
 *  to an ostream. However, an ostream has additional state that controls its
 *  behavior, such as its formatting attributes, error state, extensible arrays, *  and registered callbacks. If these are modified during the computation, the *  reducer implementation cannot guarantee that they will be the same in a
 *  parallel computation as in a serial computation. In particular:
 *
 *  -   In the serial execution, the ostream state in the continuation of a
 *      spawn will be the same as the state at the end of the spawned function.
 *      In the parallel execution, if the continuation is stolen, its view will
 *      contain a newly created ostream with the default initial state.
 *  -   In the serial execution, the ostream state following a sync is the same
 *      as the state before the sync. In the parallel execution, if the
 *      continuation is stolen, then the state following the sync will be the
 *      same as the state at the end of some spawned function.
 *
 *  In short, you must not make any assumptions about the stream state of an
 *  ostream reducer:
 *
 *  -   Following a `cilk_spawn`.
 *  -   Following a `cilk_sync`.
 *  -   At the start of an iteration of a `cilk_for` loop.
 *  -   Following the completion of a `cilk_for` loop.
 *
 *  @section redostream_types Type and Operator Requirements
 *
 *  `std::basic_ostream<Char, Traits>` must be a valid type.
*/

namespace cilk {

/** @ingroup ReducersOstream */
//@{

/** The ostream reducer view class.
 *
 *  This is the view class for reducers created with
 *  `cilk::reducer< cilk::op_basic_ostream<Char, Traits> >`. It holds the
 *  actual ostream for a parallel strand, and allows only stream output
 *  operations to be performed on it.
 *
 *  @note   The reducer "dereference" operation (`reducer::operator *()`)
 *          yields a reference to the view. Thus, for example, the view
 *          class's `<<` operation would be used in an expression like
 *          `*r << "x = " << x`, where `r` is an ostream reducer.
 *
 *  @tparam Char        The ostream element type (not the ostream type).
 *  @tparam Traits      The character traits type.
 *
 *  @see ReducersOstream
 *  @see op_basic_ostream
 */
template<typename Char, typename Traits>
class op_basic_ostream_view : public std::basic_ostream<Char, Traits>
{
    typedef std::basic_ostream<Char, Traits>  base;
    typedef std::basic_ostream<Char, Traits>  ostream_type;

    // A non-leftmost view is associated with a private string buffer. (The
    // leftmost view is associated with the buffer of the reducer's associated
    // ostream, so its private buffer is unused.)
    //
    std::basic_stringbuf<Char, Traits> m_buffer;

public:

    /** Value type. Required by @ref monoid_with_view.
     */
    typedef ostream_type value_type;

    /** Reduce operation. Required by @ref monoid_with_view.
     */
    void reduce(op_basic_ostream_view* other)
    {
        // Writing an empty buffer results in failure. Testing `sgetc()` is the
        // easiest way of checking for an empty buffer.
        if (other->m_buffer.sgetc() != Traits::eof()) {
            *this << (&other->m_buffer);
        }
    }

    /** Non-leftmost (identity) view constructor. The view is associated with
     *  its internal buffer. Required by @ref monoid_base.
     */
    op_basic_ostream_view() : base(&m_buffer) {}

    /** Leftmost view constructor. The view is associated with an existing
     *  ostream.
     */
    op_basic_ostream_view(const ostream_type& os) : base(0)
    {
        base::rdbuf(os.rdbuf());       // Copy stream buffer
        base::flags(os.flags());       // Copy formatting flags
        base::setstate(os.rdstate());  // Copy error state
    }

    /** Sets/gets.
     *
     *  These are all no-ops.
     */
    //@{

    void view_set_value(const value_type&)
        { assert("set_value() is not allowed on ostream reducers" && 0); }
    const value_type& view_get_value() const
        { assert("get_value() is not allowed on ostream reducers" && 0);
          return *this; }
    typedef value_type const& return_type_for_get_value;
    void view_move_in(const value_type&)
        { assert("move_in() is not allowed on ostream reducers" && 0); }
    void view_move_out(const value_type&)
        { assert("move_out() is not allowed on ostream reducers" && 0); }

    //@}
};

/** Ostream monoid class. Instantiate the cilk::reducer template class with an
 *  op_basic_ostream monoid to create an ostream reducer class:
 *
 *      cilk::reducer< cilk::op_basic_string<char> > r;
 *
 *  @tparam Char        The stream element type (not the stream type).
 *  @tparam Traits      The character traits type.
 *
 *  @see ReducersOstream
 *  @see op_basic_ostream_view
 *  @see reducer_ostream
 *  @see op_ostream
 *  @see op_wostream
 */
template<typename Char,
         typename Traits = std::char_traits<Char>,
         bool     Align = false>
class op_basic_ostream :
    public monoid_with_view< op_basic_ostream_view<Char, Traits>, Align >
{
    typedef monoid_with_view< op_basic_ostream_view<Char, Traits>, Align >
            base;
    typedef std::basic_ostream<Char, Traits>            ostream_type;
    typedef provisional_guard<typename base::view_type> view_guard;

public:

    /** View type of the monoid.
     */
    typedef typename base::view_type view_type;

    /** @name Construct function.
     *
     *  The only supported ostream reducer constructor takes a reference to
     *  an existing ostream.
     *
     *  @param os   The ostream destination for receive all data written to the
     *              reducer.
     */
    static void construct(
        op_basic_ostream*   monoid,
        view_type*          view,
        const ostream_type& os)
    {
        view_guard vg( new((void*) view) view_type(os) );
        vg.confirm_if( new((void*) monoid) op_basic_ostream );
    }
};


/**
 *  Convenience typedef for narrow ostreams.
 */
typedef op_basic_ostream<char> op_ostream;

/**
 *  Convenience typedef for wide ostreams.
 */
typedef op_basic_ostream<wchar_t> op_wostream;

/// @cond internal

class reducer_ostream;

/** Metafunction specialization for reducer conversion.
 *
 *  This specialization of the @ref legacy_reducer_downcast template class
 *  defined in reducer.h causes the `reducer<op_basic_ostream<char> >` class
 *  to have an `operator reducer_ostream& ()` conversion operator that
 *  statically downcasts the `reducer<op_basic_ostream<char> >` to
 *  `reducer_ostream`. (The reverse conversion, from `reducer_ostream` to
 *  `reducer<op_basic_ostream<char> >`, is just an upcast, which is provided
 *  for free by the language.)
 */
template<bool Align>
struct legacy_reducer_downcast<
    reducer<op_basic_ostream<char, std::char_traits<char>, Align> > >
{
    typedef reducer_ostream type;
};

/// @endcond

/** Deprecated ostream reducer class.
 *
 *  reducer_ostream is the same as @ref cilk::reducer<@ref op_ostream>, except
 *  that reducer_ostream is a proxy for the contained view, so that ostream
 *  operations can be applied directly to the reducer. For example, a number is
 *  written to a `reducer<op_ostream>` with `*r << x`, but a number can be
 *  written to a `reducer_ostream` with `r << x`.
 *
 *  @deprecated Users are strongly encouraged to use `reducer<monoid>`
 *              reducers rather than the old wrappers like reducer_ostream. The
 *              `reducer<monoid>` reducers show the reducer/monoid/view
 *              architecture more clearly, are more consistent in their
 *              implementation, and present a simpler model for new
 *              user-implemented reducers.
 *
 *  @note   Implicit conversions are provided between `%reducer_ostream`
 *          and `reducer<%op_ostream>`. This allows incremental code
 *          conversion: old code that used  `%reducer_ostream` can pass a
 *          `%reducer_ostream` to a converted function that now expects a
 *          pointer or reference to a `reducer<%op_ostream>`, and vice versa.
 *
 *  @tparam Char        The stream element type (not the stream type).
 *  @tparam Traits      The character traits type.
 *
 *  @see op_ostream
 *  @see reducer
 *  @see ReducersOstream
 */
class reducer_ostream :
      public reducer<op_basic_ostream<char, std::char_traits<char>, true> >
{
    typedef reducer<op_basic_ostream<char, std::char_traits<char>, true> > base;
    using base::view;
public:

    /// The view type for the reducer.
    typedef base::view_type        View;

    /// The monoid type for the reducer.
    typedef base::monoid_type      Monoid;

    /** Constructs an initial `reducer_ostream` from a `std::ostream`.  The
     *  specified stream is used as the eventual destination for all text
     *  streamed to this hyperobject.
     */
    explicit reducer_ostream(const std::ostream &os) : base(os) {}

    /** Returns a modifiable reference to the underlying 'ostream' object.
     */
    std::ostream& get_reference() { return view(); }

    /** Writes to the ostream.
     */
    template<typename T>
    std::ostream& operator<< (const T &v)
    {
        return view() << v;
    }

    /**
     * Calls a manipulator.
     *
     * @param _Pfn Pointer to the manipulator function.
     */
    reducer_ostream& operator<< (std::ostream &(*_Pfn)(std::ostream &))
    {
        (*_Pfn)(view());
        return *this;
    }

    /** @name Dereference
     *  @details Dereferencing a wrapper is a no-op. It simply returns the
     *  wrapper. Combined with the rule that the wrapper forwards view
     *  operations to its contained view, this means that view operations can
     *  be written the same way on reducers and wrappers, which is convenient
     *  for incrementally converting old code using wrappers to use reducers
     *  instead. That is:
     *
     *      reducer<op_ostream> r;
     *      *r << "a";      // *r returns the view
     *                      // operator<<() is a view member function
     *
     *      reducer_ostream w;
     *      *w << "a";      // *w returns the wrapper
     *                      // operator<<() is a wrapper member function
     *                      // that calls the corresponding view function
     */
    //@{
    reducer_ostream&       operator*()       { return *this; }
    reducer_ostream const& operator*() const { return *this; }

    reducer_ostream*       operator->()       { return this; }
    reducer_ostream const* operator->() const { return this; }
    //@}

    /** @name Upcast
     *  @details In Intel Cilk Plus library 0.9, reducers were always cache-aligned.
     *  In library  1.0, reducer cache alignment is optional. By default,
     *  reducers are unaligned (i.e., just naturally aligned), but legacy
     *  wrappers inherit from cache-aligned reducers for binary compatibility.
     *
     *  This means that a wrapper will automatically be upcast to its aligned
     *  reducer base class. The following conversion operators provide
     *  pseudo-upcasts to the corresponding unaligned reducer class.
     */
    //@{
    operator reducer<op_ostream>& ()
    {
        return *reinterpret_cast< reducer<op_ostream>* >(this);
    }
    operator const reducer<op_ostream>& () const
    {
        return *reinterpret_cast< const reducer<op_ostream>* >(this);
    }
    //@}
};

} // namespace cilk

#endif // REDUCER_OSTREAM_H_INCLUDED
