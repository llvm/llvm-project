/*
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
 *
 */

/*
 * reducer_ostream.h
 *
 * Purpose: Hyper-object to write to 'std::ostream's
 *
 * Classes: reducer_ostream
 *
 * Description:
 * ============
 * Output streams ('std::ostream's) are a convenient means of writing text to
 * files, the user console, or sockets.  In a serial program, text is written
 * to an ostream in a specific, logical order.  For example, computing while
 * traversing a data structure and printing them to an 'ostream' will result
 * in the values being printed in the order of traversal.  In a parallel
 * version of the same program, however, different parts of the data structure
 * may be traversed in a different order, resulting in a non-deterministic
 * ordering of the stream.  Worse, multiple strands may write to the same
 * stream simultaneously, resulting in a data race.  Replacing the
 * 'std::ostream' with a 'cilk::reducer_ostream' will solve both problems: Data
 * will appeaer in the stream in the same order as it would for the serial
 * program, and there will be no races (no locks) on the common stream.
 *
 * Usage Example:
 * ==============
 * Assume we wish to traverse an array of objects, performing an operation on
 * each object and writing the result to a file.  Without a reducer_ostream,
 * we have a race on the 'output' file stream:
 *..
 *  void compute(std::ostream& os, double x)
 *  {
 *      // Perform some significant computation and print the result:
 *      os << std::asin(x);
 *  }
 *
 *  int test()
 *  {
 *      const std::size_t ARRAY_SIZE = 1000000;
 *      extern double myArray[ARRAY_SIZE];
 *
 *      std::ofstream output("output.txt");
 *      cilk_for (std::size_t i = 0; i < ARRAY_SIZE; ++i)
 *      {
 *          compute(output, myArray[i]);
 *      }
 *
 *      return 0;
 *  }
 *..
 * The race is solved by using a reducer_ostream to proxy the 'output' file:
 *..
 *  void compute(cilk::reducer_ostream& os, double x)
 *  {
 *      // Perform some significant computation and print the result:
 *      *os << std::asin(x);
 *  }
 *
 *  int test()
 *  {
 *      const std::size_t ARRAY_SIZE = 1000000;
 *      extern double myArray[ARRAY_SIZE];
 *
 *      std::ofstream output("output.txt");
 *      cilk::reducer_ostream hyper_output(output);
 *      cilk_for (std::size_t i = 0; i < ARRAY_SIZE; ++i)
 *      {
 *          compute(hyper_output, myArray[i]);
 *      }
 *
 *      return 0;
 *  }
 *..
 *
 * Limitations:
 * ============
 * There are two possible values for the formatting flags immediately after a
 * 'cilk_spawn' statement: they may either have the value that was set by the
 * spawn function, or they may have default values.  Because of
 * non-determinism in the processor scheduling, there is no way to determine
 * which it will be.  Similarly, the formatting flags after a 'cilk_sync' may
 * or may not have the same value as before the sync.  Therefore, one must use
 * a disciplined coding style to avoid formatting errors.  There are two
 * approaches to mitigating the problem: The first is to eliminate the
 * difference between the two possible outcomes by ensuring that the spawned
 * function always returns the flags to their initial state:
 *..
 *  void compute(cilk::reducer_ostream& os, double x)
 *  {
 *      // Perform some significant computation and print the result:
 *      int saveprec = os.precision(5);
 *      os << std::asin(x);
 *      os.precision(saveprec);
 *  }
 *..
 * The second approach is to write your streaming operations such that they
 * don't depend on the previous state of the formatting flags by setting any
 * important flags before every block of output:
 *..
 *      cilk_spawn compute(hyper_output, value);
 *
 *      hyper_output->precision(2);  // Don't depend on previous precision
 *      *hyper_output << f();
 *      *hyper_output << g();
 *..
 * Another concern is memory usage.  A reducer_ostream will buffer as much text
 * as necessary to ensure that the order of output matches that of the serial
 * version of the program.  If all spawn branches perform an equal amount of
 * output, then one can expect that half of the output before a sync will be
 * buffered in memory.  This hyperobject is therefore not well suited for
 * serializing very large quantities of text output.
 */

#ifndef REDUCER_OSTREAM_H_INCLUDED
#define REDUCER_OSTREAM_H_INCLUDED

#include <cilk/reducer.h>
#include <iostream>
#include <sstream>

namespace cilk {

/**
 * @brief Class 'reducer_ostream' is the representation of a hyperobject for
 * output text streaming.
 */
class reducer_ostream
{
public:
    /// Internal representation of the per-strand view of the data for reducer_ostream
    class View: public std::ostream
    {
    public:
        /// Type of the std::stream reducer_ostream is based on
        typedef std::ostream Base;

        friend class reducer_ostream;

        View():
            std::ostream(0)
        {
            Base::rdbuf(&strbuf_);
        };

    private:
        void use_ostream (const std::ostream &os)
        {
            Base::rdbuf(os.rdbuf());
            Base::flags(os.flags());       // Copy formatting flags
            Base::setstate(os.rdstate());  // Copy error state
        }

    private:
        std::stringbuf  strbuf_;
    };

public:
    /// Definition of data view, operation, and identity for reducer_ostream
    struct Monoid: monoid_base< View >
    {
        static void reduce (View *left, View *right);
    };

private:
    // Hyperobject to serve up views
    reducer<Monoid> imp_;

    // Methods that provide the API for the reducer
public:

    // Construct an initial 'reducer_ostream' from an 'std::ostream'.  The
    // specified 'os' stream is used as the eventual destination for all
    // text streamed to this hyperobject.
    explicit reducer_ostream(const std::ostream &os);

    // Return a modifiable reference to the underlying 'ostream' object.
    std::ostream& get_reference();

    /**
     * Append data from some type to the reducer_ostream
     *
     * @param v Value to be appended to the reducer_ostream
     */
    template<typename T>
    std::ostream &
    operator<< (const T &v)
    {
        return imp_.view() << v;
    }

    /**
     * Append data from a std::ostream to the reducer_ostream
     *
     * @param _Pfn std::ostream to copy from
     */
    std::ostream &
    operator<< (std::ostream &(*_Pfn)(std::ostream &))
    {
        View &v = imp_.view();

        return ((*_Pfn)(v));
    }

    reducer_ostream&       operator*()       { return *this; }
    reducer_ostream const& operator*() const { return *this; }

    reducer_ostream*       operator->()       { return this; }
    reducer_ostream const* operator->() const { return this; }
};


// -------------------------------------------
// class reducer_ostream::Monoid
// -------------------------------------------

/**
 * Appends string from "right" reducer_basic_string onto the end of
 * the "left". When done, the "right" reducer_basic_string is empty.
 */
void
reducer_ostream::Monoid::reduce(View *left, View *right)
{
    left->operator<< (&right->strbuf_);
}

// --------------------------
// class reducer_ostream
// --------------------------

/**
 * Construct a reducer_ostream which will write to the specified std::ostream
 *
 * @param os std::ostream to write to
 */
inline
reducer_ostream::reducer_ostream(const std::ostream &os) :
    imp_()
{
    View &v = imp_.view();

    v.use_ostream(os);
}

/**
 * Get a reference to the std::ostream
 */
inline
std::ostream &
reducer_ostream::get_reference()
{
    View &v = imp_.view();

    return v;
}

} // namespace cilk

#endif //  REDUCER_OSTREAM_H_INCLUDED

