/*
 *  @copyright
 *  Copyright (C) 2011-2013, Intel Corporation
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
 * holder.h
 *
 * Purpose: hyperobject to provide different views of an object to each
 * parallel strand.
 */

#ifndef HOLDER_H_INCLUDED
#define HOLDER_H_INCLUDED

#include <cilk/reducer.h>
#include <memory>
#include <utility>

#ifdef __cplusplus

/* C++ Interface
 *
 * Classes: holder<Type>
 *
 * Description:
 * ============
 * This component provides a hyperobject that isolates a parallel uses of a
 * common variable where it is not necessary to preserve changes from
 * different parallel strands.  In effect, a holder acts a bit like
 * thread-local storage, but has qualities that work better with the
 * fork-join structure of Cilk.  In particular, a holder has the following
 * qualities:
 *
 * - The view of a holder before the first spawn within a function is the same
 *   as the view after each sync (as in the case of a reducer).
 * - The view of a holder within the first spawned child of a function (or the
 *   first child spawned after a sync) is the same as the view on entry to the
 *   function.
 * - The view of a holder before entering a _Cilk_for loop is the same as the
 *   view during the first iteration of the loop and the view at the end of
 *   the loop.
 * - The view of a holder in the continuation of a spawn or in an arbitrary
 *   iteration of a _Cilk_for loop is *non-deterministic*.  It is generally
 *   recommended that the holder be explicitly put into a known state in these
 *   situations.
 *
 * A holder can be used as an alternative to parameter-passing.  They are most
 * useful for replacing non-local variables without massive refactoring.  A
 * holder takes advantage of the fact that, most of the time, a holder view
 * does not change after a spawn or from one iteration of a parallel for loop
 * to the next (i.e., stealing is the exception, not the rule).  When the
 * holder view is a large object that is expensive to construct, this
 * optimization can save significant time versus creating a separate local
 * object for each view.  In addition, a holder using the "keep last" policy
 * will have the same value after a sync as the serialization of the same
 * program.  The last quality will often allow the program to avoid
 * recomputing a value.
 *
 * Usage Example:
 * ==============
 * Function 'compute()' is a complex function that computes a value using a
 * memoized algorithm, storing intermediate results in a hash table.  Compute
 * calls several other functions, each of which calls several other functions,
 * all of which share a global hash table.  In all, there are over a dozen
 * functions with a total of about 60 references to the hash table.  
 *..
 *  hash_table<int, X> memos;
 *
 *  void h(const X& x);  // Uses memos
 *
 *  double compute(const X& x)
 *  {
 *     memos.clear();
 *     // ...
 *     memos[i] = x;
 *     ...
 *     g(i);  // Uses memos
 *     // ...
 *     std::for_each(c.begin(), c.end(), h);  // Call h for each element of c
 *  }
 *
 *  int main()
 *  {
 *      const std::size_t ARRAY_SIZE = 1000000;
 *      extern X myArray[ARRAY_SIZE];
 *
 *      for (std::size_t i = 0; i < ARRAY_SIZE; ++i)
 *      {
 *          compute(myArray[i]);
 *      }
 *  }
 *..
 * We would like to replace the 'for' loop in 'main' with a 'cilk_for'.
 * Although the hash table is cleared on entry to each call to 'compute()',
 * and although the values stored in the hash table are no longer used after
 * 'compute()' returns, the use of the hash table as a global variable
 * prevents 'compute()' from being called safely in parallel.  One way to do
 * this would be to make 'memos' a private variable within the cilk_for loop
 * and pass it down to the actual computation, so that each loop iteration has
 * its own private copy:
 *..
 *      cilk_for (std::size_t i = 0; i < ARRAY_SIZE; ++i)
 *      {
 *          hash_table<int, X> memos;
 *          compute(myArray[i], memos);
 *      }
 *..
 * The problem with this approach is that it requires changing the signature
 * of 'compute', 'h', 'g', and every one of the dozen or so functions that
 * reference 'memos' as well as any function that calls those functions.  This
 * may break the abstraction of 'compute' and other functions, exposing an
 * implementation detail that was not part of the interface.  In addition, the
 * function 'h' is called through a templated algorithm, 'for_each', which
 * requires a fixed interface.  Finally, there is constructor and destructor
 * overhead for 'hash_table' each time through the loop.
 *
 * The alternative approach is to replace 'memos' with a holder.  The holder
 * would be available to all of the functions involved, but would not cause a
 * race between parallel loop iterations.  In order to make this work, each
 * use of the 'memos' variable must be (mechanically) replaced by a use of the
 * holder:
 *..
 *  cilk::holder<hash_table<int, X> > memos_h;
 *
 *  void h(const X& x);  // Uses memos_h
 *
 *  double compute(const X& x)
 *  {
 *     memos_h().clear();  // operator() used to "dereference" the holder
 *     // ...
 *     memos_h()[i] = x;   // operator() used to "dereference" the holder
 *     ...
 *     g(i);  // Uses memos_h
 *     // ...
 *     std::for_each(c.begin(), c.end(), h);  // Call h for each element of c
 *  }
 *..
 * Note that each reference to the holder must be modified with an empty pair
 * of parenthesis.  This syntax is needed because there is no facility in C++
 * for a "smart reference" that would allow 'memos_h' to be a perfect
 * replacement for 'memos'.  One way that a user can avoid this syntax change
 * is to wrap the holder in a class that has the same inteface as
 * 'hash_table' but redirects all calls to the holder:
 *..
 *  template <typename K, typename V>
 *  class hash_table_holder
 *  {
 *    private:
 *      cilk::holder<hash_table<K, V> > m_holder;
 *    public:
 *      void clear() { m_holder().clear(); }
 *      V& operator[](const K& x) { return m_holder()[x]; }
 *      std::size_t size() const { return m_holder().size(); }
 *      // etc. ...
 *  };
 *..
 * Using the above wrapper, the original code can be left unchanged except for
 * replacing 'hash_table' with 'hash_table_holder' and replacing 'for' with
 * 'cilk_for':
 *..
 *  hash_table_holder<int, X> memos;
 *
 *  void h(const X& x);  // Uses memos
 *
 *  double compute(const X& x)
 *  {
 *     memos.clear();  // Calls hash_table_holder::clear().
 *     // ...
 *  }
 *..
 * The above changes have no benefit over the use of thread-local storage.
 * What if one of the functions has a 'cilk_spawn', however?
 *..
 *  void h(const X& x)
 *  {
 *      Y y = x.nested();
 *      double d, w;
 *      if (y)
 *      {
 *          w = cilk_spawn compute_width(y); // May use 'memos'
 *          d = compute_depth(y);            // Does not use 'memos'
 *          cilk_sync;
 *          compute(y);  // recursive call.  Uses 'memos'.
 *      }
 *  }
 *..
 * In the above example, the view of the holder within 'compute_width' is the
 * same as the view on entry to 'h'.  More importantly, the view of the holder
 * within the recursive call to 'compute' is the same as the view on entry to
 * 'h', even if a different worker is executing the recursive call.  Thus, the
 * holder view within a Cilk program has useful qualities not found in
 * thread-local storage.
 */

namespace cilk {
    
    /**
     * After a sync, the value stored in a holder matches the most recent
     * value stored into the holder by one of the starnds entering the sync.
     * The holder policy used to instantiate the holder determines which of
     * the entering strands determines the final value of the holder. A policy
     * of 'holder_keep_indeterminate' (the default) is the most efficient, and
     * results in an indeterminate value depending on the runtime schedule
     * (see below for more specifics).  An indeterminate value after a sync is
     * often acceptable, especially if the value of the holder is not reused
     * after the sync.  All of the remaining policies retain the value of the
     * last strand that would be executed in the serialization of the program.
     * They differ in the mechanism used to move the value from one view to
     * another.  A policy of 'holder_keep_last_copy' moves values by
     * copy-assignment.  A policy of 'holder_keep_last_swap' moves values by
     * calling 'swap'.  A policy of 'holder_keep_last_move' is available only
     * for compilers that support C++0x rvalue references and moves values by
     * move-assignment.  A policy of 'holder_keep_last' attempts to choose the
     * most efficient mechanism: member-function 'swap' if the view type
     * supports it, otherwise move-assignment if supported, otherwise
     * copy-assignment.  (The swap member function for a class that provides
     * one is almost always as fast or faster than move-assignment or
     * copy-assignment.)
     *
     * The behavior of 'holder_keep_indeterminate', while indeterminate, is
     * not random and can be used for advanced programming or debugging.  With
     * a policy of 'holder_keep_intermediate', values are never copied or
     * moved between views.  The value of the view after a sync is the same as
     * the value set in the last spawned child before a steal occurs or the
     * last value set in the continuation if no steal occurs.  Using this
     * knowledge, a programmer can use a holder to detect the earliest steal
     * in a piece of code.  An indeterminate holder is also useful for keeping
     * cached data similar to the way some applications might use thread-local
     * storage.
     */
    enum holder_policy {
        holder_keep_indeterminate,
        holder_keep_last,
        holder_keep_last_copy,
        holder_keep_last_swap,
#ifdef __CILKRTS_RVALUE_REFERENCES
        holder_keep_last_move
#endif
    };

    namespace internal {

        // Private special-case holder policy using the swap member-function
        const holder_policy holder_keep_last_member_swap =
            (holder_policy) (holder_keep_last_swap | 0x10);

        /* The constant, 'has_member_swap<T>::value', will be 'true' if 'T'
         * has a non-static member function with prototype 'void swap(T&)'.
         * The mechanism used to detect 'swap' is the most portable among
         * present-day compilers, but is not the most robust.  Specifically,
         * the prototype for 'swap' must exactly match 'void swap(T&)'.
         * Near-matches like a 'swap' function that returns 'int' instead of
         * 'void' will not be detected.  Detection will also fail if 'T'
         * inherits 'swap' from a base class.
         */
        template <typename T>
        class has_member_swap
        {
            // This technique for detecting member functions was described by
            // Rani Sharoni in comp.lang.c++.moderated:
            // http://groups.google.com/group/comp.lang.c++.moderated/msg/2b06b2432fddfb60

            // sizeof(notchar) is guaranteed larger than 1
            struct notchar { char x[2]; };

            // Instantiationg Q<U, &U::swap> will fail unless U contains a
            // non-static member with prototype 'void swap(U&)'.
            template <class U, void (U::*)(U&)> struct Q { };

            // First 'test' is preferred overload if U::swap exists with the
            // correct prototype.  Second 'test' is preferred overload
            // otherwise.
            template <typename U> static char test(Q<U,&U::swap>*);
            template <typename U> static notchar test(...);

        public:
            /// 'value' will be true if T has a non-static member function
            /// with prototype 'void swap(T&)'.
            static const bool value = (1 == sizeof(test<T>(0)));
        };

        template <typename T> const bool has_member_swap<T>::value;

        /**
         * @brief Utility class for exception safety.
         *
         * The constuctor for this class takes a pointer and an allocator and
         * holds on to them.  The destructor deallocates the pointed-to
         * object, without calling its destructor, typically to recover memory
         * in case an exception is thrown. The release member clears the
         * pointer so that the deallocation is prevented, i.e., when the
         * exception danger has passed.  The behavior of this class is similar
         * to auto_ptr and unique_ptr.
         */
        template <typename Type, typename Allocator = std::allocator<Type> >
        class auto_deallocator
        {
            Allocator m_alloc;
            Type*     m_ptr;

            // Non-copiable
            auto_deallocator(const auto_deallocator&);
            auto_deallocator& operator=(const auto_deallocator&);

        public:
            /// Constructor
            explicit auto_deallocator(Type* p, const Allocator& a = Allocator())
                : m_alloc(a), m_ptr(p) { }

            /// Destructor - free allocated resources
            ~auto_deallocator() { if (m_ptr) m_alloc.deallocate(m_ptr, 1); }

            /// Remove reference to resource
            void release() { m_ptr = 0; }
        };

        /**
         * Pure-abstract base class to initialize holder views
         */
        template <typename Type, typename Allocator>
        class init_base
        {
        public:
            virtual ~init_base() { }
            virtual init_base* clone_self(Allocator& a) const = 0;
            virtual void delete_self(Allocator& a) = 0;
            virtual void construct_view(Type* p, Allocator& a) const = 0;
        };

        /**
         * Class to default-initialize a holder view
         */
        template <typename Type, typename Allocator>
        class default_init : public init_base<Type, Allocator>
        {
            typedef init_base<Type, Allocator> base;

            /// Private constructor (called from static make() function).
            default_init() { }

            // Non-copiable
            default_init(const default_init&);
            default_init& operator=(const default_init&);

        public:
            // Static factory function
            static default_init* make(Allocator& a);

            // Virtual function overrides
            virtual ~default_init();
            virtual base* clone_self(Allocator& a) const;
            virtual void delete_self(Allocator& a);
            virtual void construct_view(Type* p, Allocator& a) const;
        };

        template <typename Type, typename Allocator>
        default_init<Type, Allocator>*
        default_init<Type, Allocator>::make(Allocator&)
        {
            // Return a pointer to a singleton.  All instances of this class
            // are identical, so we need only one.
            static default_init self;
            return &self;
        }

        template <typename Type, typename Allocator>
        default_init<Type, Allocator>::~default_init()
        {
        }

        template <typename Type, typename Allocator>
        init_base<Type, Allocator>*
        default_init<Type, Allocator>::clone_self(Allocator& a) const
        {
            return make(a);
        }

        template <typename Type, typename Allocator>
        void default_init<Type, Allocator>::delete_self(Allocator&)
        {
            // Since make() returned a shared singleton, there is nothing to
            // delete here.
        }

        template <typename Type, typename Allocator>
        void
        default_init<Type, Allocator>::construct_view(Type* p,
                                                      Allocator&) const
        {
            ::new((void*) p) Type();
            // TBD: In a C++0x library, this should be rewritten
            // std::allocator_traits<Allocator>::construct(a, p);
        }

        /**
         * Class to copy-construct a view from a stored exemplar.
         */
        template <typename Type, typename Allocator>
        class exemplar_init : public init_base<Type, Allocator>
        {
            typedef init_base<Type, Allocator> base;

            Type* m_exemplar;

            // Private constructors (called from make() functions).
            exemplar_init(const Type& val, Allocator& a);
#ifdef __CILKRTS_RVALUE_REFERENCES
            exemplar_init(Type&& val,      Allocator& a);
#endif

            // Non-copyiable
            exemplar_init(const exemplar_init&);
            exemplar_init& operator=(const exemplar_init&);

        public:
            // Static factory functions
            static exemplar_init* make(const Type& val,
                                       Allocator& a = Allocator());
#ifdef __CILKRTS_RVALUE_REFERENCES
            static exemplar_init* make(Type&& val,
                                       Allocator& a = Allocator());
#endif

            // Virtual function overrides
            virtual ~exemplar_init();
            virtual base* clone_self(Allocator& a) const;
            virtual void delete_self(Allocator& a);
            virtual void construct_view(Type* p, Allocator& a) const;
        };

        template <typename Type, typename Allocator>
        exemplar_init<Type, Allocator>::exemplar_init(const Type& val,
                                                      Allocator&  a)
        {
            m_exemplar = a.allocate(1);
            auto_deallocator<Type, Allocator> guard(m_exemplar, a);
            a.construct(m_exemplar, val);
            guard.release();
        }

#ifdef __CILKRTS_RVALUE_REFERENCES
        template <typename Type, typename Allocator>
        exemplar_init<Type, Allocator>::exemplar_init(Type&&     val,
                                                      Allocator& a)
        {
            m_exemplar = a.allocate(1);
            auto_deallocator<Type, Allocator> guard(m_exemplar, a);
            a.construct(m_exemplar, std::forward<Type>(val));
            guard.release();
        }
#endif

        template <typename Type, typename Allocator>
        exemplar_init<Type, Allocator>*
        exemplar_init<Type, Allocator>::make(const Type& val,
                                             Allocator&  a)
        {
            typedef typename Allocator::template rebind<exemplar_init>::other
                self_alloc_t;
            self_alloc_t alloc(a);

            exemplar_init *self = alloc.allocate(1);
            auto_deallocator<exemplar_init, self_alloc_t> guard(self, alloc);

            // Don't use allocator to construct self.  Allocator should be
            // used only on elements of type 'Type'.
            ::new((void*) self) exemplar_init(val, a);

            guard.release();

            return self;
        }

#ifdef __CILKRTS_RVALUE_REFERENCES
        template <typename Type, typename Allocator>
        exemplar_init<Type, Allocator>*
        exemplar_init<Type, Allocator>::make(Type&&           val,
                                             Allocator& a)
        {
            typedef typename Allocator::template rebind<exemplar_init>::other
                self_alloc_t;
            self_alloc_t alloc(a);

            exemplar_init *self = alloc.allocate(1);
            auto_deallocator<exemplar_init, self_alloc_t> guard(self, alloc);

            // Don't use allocator to construct self.  Allocator should be
            // used only on elements of type 'Type'.
            ::new((void*) self) exemplar_init(std::forward<Type>(val), a);

            guard.release();

            return self;
        }
#endif

        template <typename Type, typename Allocator>
        exemplar_init<Type, Allocator>::~exemplar_init()
        {
            // Called only by delete_self, which deleted the exemplar using an
            // allocator.
            __CILKRTS_ASSERT(0 == m_exemplar);
        }

        template <typename Type, typename Allocator>
        init_base<Type, Allocator>*
        exemplar_init<Type, Allocator>::clone_self(Allocator& a) const
        {
            return make(*m_exemplar, a);
        }

        template <typename Type, typename Allocator>
        void exemplar_init<Type, Allocator>::delete_self(Allocator& a)
        {
            typename Allocator::template rebind<exemplar_init>::other alloc(a);

            a.destroy(m_exemplar);
            a.deallocate(m_exemplar, 1);
            m_exemplar = 0;

            this->~exemplar_init();
            alloc.deallocate(this, 1);
        }

        template <typename Type, typename Allocator>
        void
        exemplar_init<Type, Allocator>::construct_view(Type*            p,
                                                       Allocator& a) const
        {
            a.construct(p, *m_exemplar);
            // TBD: In a C++0x library, this should be rewritten
            // std::allocator_traits<Allocator>::construct(a, p, *m_exemplar);
        }

        /**
         * Class to construct a view using a stored functor.  The functor,
         * 'f', must be be invokable using the expression 'Type x = f()'.
         */
        template <typename Func, typename Allocator>
        class functor_init :
            public init_base<typename Allocator::value_type, Allocator>
        {
            typedef typename Allocator::value_type            value_type;
            typedef init_base<value_type, Allocator>          base;
            typedef typename Allocator::template rebind<Func>::other f_alloc;

            Func *m_functor;

            /// Private constructors (called from make() functions
            functor_init(const Func& f, Allocator& a);
#ifdef __CILKRTS_RVALUE_REFERENCES
            functor_init(Func&& f, Allocator& a);
#endif

            // Non-copiable
            functor_init(const functor_init&);
            functor_init& operator=(const functor_init&);

        public:
            // Static factory functions
            static functor_init* make(const Func& val,
                                      Allocator& a = Allocator());
#ifdef __CILKRTS_RVALUE_REFERENCES
            static functor_init* make(Func&& val,
                                      Allocator& a = Allocator());
#endif

            // Virtual function overrides
            virtual ~functor_init();
            virtual base* clone_self(Allocator& a) const;
            virtual void delete_self(Allocator& a);
            virtual void
                construct_view(value_type* p, Allocator& a) const;
        };

        /// Specialization to strip off reference from 'Func&'.
        template <typename Func, typename Allocator>
        struct functor_init<Func&, Allocator>
            : functor_init<Func, Allocator> { };

        /// Specialization to strip off reference and cvq from 'const Func&'.
        template <typename Func, typename Allocator>
        struct functor_init<const Func&, Allocator>
            : functor_init<Func, Allocator> { };

        template <typename Func, typename Allocator>
        functor_init<Func, Allocator>::functor_init(const Func& f,
                                                    Allocator&  a)
        {
            f_alloc alloc(a);

            m_functor = alloc.allocate(1);
            auto_deallocator<Func, f_alloc> guard(m_functor, alloc);
            alloc.construct(m_functor, f);
            guard.release();
        }

#ifdef __CILKRTS_RVALUE_REFERENCES
        template <typename Func, typename Allocator>
        functor_init<Func, Allocator>::functor_init(Func&&     f,
                                                    Allocator& a)
        {
            f_alloc alloc(a);

            m_functor = alloc.allocate(1);
            auto_deallocator<Func, f_alloc> guard(m_functor, alloc);
            alloc.construct(m_functor, std::forward<Func>(f));
            guard.release();
        }
#endif

        template <typename Func, typename Allocator>
        functor_init<Func, Allocator>*
        functor_init<Func, Allocator>::make(const Func& f, Allocator& a)
        {
            typedef typename Allocator::template rebind<functor_init>::other
                self_alloc_t;
            self_alloc_t alloc(a);

            functor_init *self = alloc.allocate(1);
            auto_deallocator<functor_init, self_alloc_t> guard(self, alloc);

            // Don't use allocator to construct self.  Allocator should be
            // used only on elements of type 'Func'.
            ::new((void*) self) functor_init(f, a);

            guard.release();

            return self;
        }

#ifdef __CILKRTS_RVALUE_REFERENCES
        template <typename Func, typename Allocator>
        functor_init<Func, Allocator>*
        functor_init<Func, Allocator>::make(Func&& f, Allocator& a)
        {
            typedef typename Allocator::template rebind<functor_init>::other
                self_alloc_t;
            self_alloc_t alloc(a);

            functor_init *self = alloc.allocate(1);
            auto_deallocator<functor_init, self_alloc_t> guard(self, alloc);

            // Don't use allocator to construct self.  Allocator should be
            // used only on elements of type 'Func'.
            ::new((void*) self) functor_init(std::forward<Func>(f), a);

            guard.release();

            return self;
        }
#endif

        template <typename Func, typename Allocator>
        functor_init<Func, Allocator>::~functor_init()
        {
            // Called only by delete_self, which deleted the functor using an
            // allocator.
            __CILKRTS_ASSERT(0 == m_functor);
        }

        template <typename Func, typename Allocator>
        init_base<typename Allocator::value_type, Allocator>*
        functor_init<Func, Allocator>::clone_self(Allocator& a) const
        {
            return make(*m_functor, a);
        }

        template <typename Func, typename Allocator>
        inline
        void functor_init<Func, Allocator>::delete_self(Allocator& a)
        {
            typename Allocator::template rebind<functor_init>::other alloc(a);
            f_alloc fa(a);

            fa.destroy(m_functor);
            fa.deallocate(m_functor, 1);
            m_functor = 0;

            this->~functor_init();
            alloc.deallocate(this, 1);
        }

        template <typename Func, typename Allocator>
        void functor_init<Func, Allocator>::construct_view(value_type* p,
                                                           Allocator& a) const
        {
            a.construct(p, (*m_functor)());
            // In C++0x, the above should be written
            // std::allocator_traits<Allocator>::construct(a, p, m_functor());
        }

        /**
         * Functor called to reduce a holder
         */
        template <typename Type, holder_policy Policy>
        struct holder_reduce_functor;

        /**
         * Specialization to keep the left (first) value.
         */
        template <typename Type>
        struct holder_reduce_functor<Type, holder_keep_indeterminate>
        {
            void operator()(Type* left, Type* right) const { }
        };

        /**
         * Specialization to copy-assign from the right (last) value.
         */
        template <typename Type>
        struct holder_reduce_functor<Type, holder_keep_last_copy>
        {
            void operator()(Type* left, Type* right) const {
                *left = *right;
            }
        };

        /*
         * Specialization to keep the right (last) value via swap.
         */
        template <typename Type>
        struct holder_reduce_functor<Type, holder_keep_last_swap>
        {
            void operator()(Type* left, Type* right) const {
                using std::swap;
                swap(*left, *right);
            }
        };

#ifdef __CILKRTS_RVALUE_REFERENCES
        /*
         * Specialization to move-assign from the right (last) value.
         */
        template <typename Type>
        struct holder_reduce_functor<Type, holder_keep_last_move>
        {
            void operator()(Type* left, Type* right) const {
                *left = std::move(*right);
            }
        };
#endif

        /*
         * Specialization to keep the right (last) value via the swap member
         * function.
         */
        template <typename Type>
        struct holder_reduce_functor<Type, holder_keep_last_member_swap>
        {
            void operator()(Type* left, Type* right) const {
                left->swap(*right);
            }
        };

        /*
         * Specialization to keep the right (last) value by the most efficient
         * means detectable.
         */
        template <typename Type>
        struct holder_reduce_functor<Type, holder_keep_last> :
            holder_reduce_functor<Type,
                                  (holder_policy)
                                  (has_member_swap<Type>::value ?
                                  holder_keep_last_member_swap :
#ifdef __CILKRTS_RVALUE_REFERENCES
                                  holder_keep_last_move
#else
                                  holder_keep_last_copy
#endif
                                  )>
        {
        };
    } // end namespace internal

    /**
     * Monoid for holders.
     * Allocator type is required to be thread-safe.
     */
    template <typename Type,
              holder_policy Policy = holder_keep_indeterminate,
              typename Allocator = std::allocator<Type> >
    class holder_monoid : public monoid_base<Type>
    {
        // Allocator is mutable because the copy of the monoid inside the
        // reducer is const (to avoid races on the shared state).  However,
        // the allocator is required to be thread-safe, so it is ok (and
        // necessary) to modify.
        mutable Allocator                     m_allocator;
        internal::init_base<Type, Allocator> *m_initializer;

    public:
        /// This constructor uses default-initialization for both the leftmost
        /// view and each identity view.
        holder_monoid(const Allocator& a = Allocator())
            : m_allocator(a)
            , m_initializer(
                internal::default_init<Type, Allocator>::make(m_allocator))
            { }

        /// These constructors use 'val' as an exemplar to copy-construct both
        /// the leftmost view and each identity view.
        holder_monoid(const Type& val, const Allocator& a = Allocator())
            : m_allocator(a)
            , m_initializer(internal::exemplar_init<Type, Allocator>::make(
                                val, m_allocator)) { }
        /// This constructor uses 'f' as a functor to construct both
        /// the leftmost view and each identity view.
        template <typename Func>
        holder_monoid(const Func& f, const Allocator& a = Allocator())
            : m_allocator(a)
            , m_initializer(
                internal::functor_init<Func, Allocator>::make(f,m_allocator))
            { }

        /// Copy constructor
        holder_monoid(const holder_monoid& rhs)
            : m_allocator(rhs.m_allocator)
            , m_initializer(rhs.m_initializer->clone_self(m_allocator)) { }

        /// "Extended" copy constructor with allocator
        holder_monoid(const holder_monoid& rhs, const Allocator& a)
            : m_allocator(a)
            , m_initializer(rhs.m_initializer->clone_self(m_allocator)) { }

#ifdef __CILKRTS_RVALUE_REFERENCES
        /// Move constructor
        holder_monoid(holder_monoid&& rhs)
            : m_allocator(rhs.m_allocator)
            , m_initializer(rhs.m_initializer) {
            rhs.m_initializer =
                internal::default_init<Type, Allocator>::make(m_allocator);
        }

        /// "Extended" move constructor with allocator
        holder_monoid(holder_monoid&& rhs, const Allocator& a)
            : m_allocator(a)
            , m_initializer(0) {
            if (a != rhs.m_allocator)
                m_initializer = rhs.m_initializer->clone_self(a);
            else {
                m_initializer = rhs.m_initializer;
                rhs.m_initializer =
                    internal::default_init<Type, Allocator>::make(m_allocator);
            }
        }
#endif
        /// Destructor
        ~holder_monoid() { m_initializer->delete_self(m_allocator); }

        holder_monoid& operator=(const holder_monoid& rhs) {
            if (this == &rhs) return *this;
            m_initializer->delete_self(m_allocator);
            m_initializer = rhs.m_initializer->clone_self(m_allocator);
        }

#ifdef __CILKRTS_RVALUE_REFERENCES
        holder_monoid& operator=(holder_monoid&& rhs) {
            if (m_allocator != rhs.m_allocator)
                // Delegate to copy-assignment on unequal allocators
                return operator=(static_cast<const holder_monoid&>(rhs));
            std::swap(m_initializer, rhs.m_initializer);
            return *this;
        }
#endif

        /// Constructs IDENTITY value into the uninitilized '*p'
        void identity(Type* p) const
            { m_initializer->construct_view(p, m_allocator); }

        /// Calls the destructor on the object pointed-to by 'p'
        void destroy(Type* p) const
            { m_allocator.destroy(p); }

        /// Return a pointer to size bytes of raw memory
        void* allocate(std::size_t s) const {
            __CILKRTS_ASSERT(sizeof(Type) == s);
            return m_allocator.allocate(1);
        }

        /// Deallocate the raw memory at p
        void deallocate(void* p) const {
            m_allocator.deallocate(static_cast<Type*>(p), sizeof(Type));
        }

        void reduce(Type* left, Type* right) const {
            internal::holder_reduce_functor<Type, Policy>()(left, right);
        }

        void swap(holder_monoid& other) {
            __CILKRTS_ASSERT(m_allocator == other.m_allocator);
            std::swap(m_initializer, other.m_initializer);
        }

        Allocator get_allocator() const {
            return m_allocator;
        }
    };

    // Namespace-scope swap
    template <typename Type, holder_policy Policy, typename Allocator>
    inline void swap(holder_monoid<Type, Policy, Allocator>& a,
                     holder_monoid<Type, Policy, Allocator>& b)
    {
        a.swap(b);
    }

   /**
    * Hyperobject to provide different views of an object to each
    * parallel strand.
    */
    template <typename Type,
              holder_policy Policy = holder_keep_indeterminate,
              typename Allocator = std::allocator<Type> >
    class holder : public reducer<holder_monoid<Type, Policy, Allocator> >
    {
        typedef holder_monoid<Type, Policy, Allocator> monoid_type;
        typedef reducer<monoid_type> imp;

        // Return a value of Type constructed using the functor Func.
        template <typename Func>
        Type make_value(const Func& f) const {
            struct obj {
                union {
                    char buf[sizeof(Type)];
                    void* align1;
                    double align2;
                };

                obj(const Func& f) { f(static_cast<Type*>(buf)); }
                ~obj() { static_cast<Type*>(buf)->~Type(); }

                operator Type&() { return *static_cast<Type*>(buf); }
            };

            return obj(f);
        }

    public:
        /// Default constructor uses default-initialization for both the
        /// leftmost view and each identity view.
        holder(const Allocator& alloc = Allocator())
            : imp(monoid_type(alloc)) { }

        /// Construct from an exemplar that is used to initialize both the
        /// leftmost view and each identity view.
        holder(const Type& v, const Allocator& alloc = Allocator())
            // Alas, cannot use an rvalue reference for 'v' because it is used
            // twice in the same expression for initializing imp.
            : imp(monoid_type(v, alloc), v) { }

        /// Construct from a functor that is used to initialize both the
        /// leftmost view and each identity view.  The functor, 'f', must be be
        /// invokable using the expression 'Type x = f()'.
        template <typename Func>
        holder(const Func& f, const Allocator& alloc = Allocator())
            // Alas, cannot use an rvalue for 'f' because it is used twice in
            // the same expression for initializing imp.
            : imp(monoid_type(f, alloc), make_value(f)) { }
    };

} // end namespace cilk

#else /* C */
# error Holders are currently available only for C++
#endif /* __cplusplus */

#endif /* HOLDER_H_INCLUDED */
