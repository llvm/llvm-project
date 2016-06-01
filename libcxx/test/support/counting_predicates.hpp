//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __COUNTING_PREDICATES_H
#define __COUNTING_PREDICATES_H


template <typename Predicate, typename Arg>
struct unary_counting_predicate : public std::unary_function<Arg, bool>  {
public:
    unary_counting_predicate(Predicate p) : p_(p), count_(0) {}
    ~unary_counting_predicate() {}

    bool operator () (const Arg &a) const { ++count_; return p_(a); }
    size_t count() const { return count_; }
    void reset() { count_ = 0; }

private:
    Predicate p_;
    mutable size_t count_;
    };


template <typename Predicate, typename Arg1, typename Arg2=Arg1>
struct binary_counting_predicate : public std::binary_function<Arg1, Arg2, bool> {
public:

    binary_counting_predicate ( Predicate p ) : p_(p), count_(0) {}
    ~binary_counting_predicate() {}

    bool operator () (const Arg1 &a1, const Arg2 &a2) const { ++count_; return p_(a1, a2); }
    size_t count() const { return count_; }
    void reset() { count_ = 0; }

private:
    Predicate p_;
    mutable size_t count_;
    };

#endif // __COUNTING_PREDICATES_H
