//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MOVEONLY_H
#define MOVEONLY_H

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES

#include <cstddef>
#include <functional>

class MoveOnly
{
    friend class MoveOnly2;
    MoveOnly(const MoveOnly&);
    MoveOnly& operator=(const MoveOnly&);

    int data_;
public:
    MoveOnly(int data = 1) : data_(data) {}
    MoveOnly(MoveOnly&& x)
        : data_(x.data_) {x.data_ = 0;}
    MoveOnly& operator=(MoveOnly&& x)
        {data_ = x.data_; x.data_ = 0; return *this;}

    int get() const {return data_;}

    bool operator==(const MoveOnly& x) const {return data_ == x.data_;}
    bool operator< (const MoveOnly& x) const {return data_ <  x.data_;}
};

namespace std {

template <>
struct hash<MoveOnly>
    : public std::unary_function<MoveOnly, std::size_t>
{
    std::size_t operator()(const MoveOnly& x) const {return x.get();}
};

}

#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

#endif  // MOVEONLY_H
