#ifndef REP_H
#define REP_H

class Rep
{
    int data_;
public:
    _LIBCPP_CONSTEXPR Rep() : data_(-1) {}
    explicit _LIBCPP_CONSTEXPR Rep(int i) : data_(i) {}

    bool _LIBCPP_CONSTEXPR operator==(int i) const {return data_ == i;}
    bool _LIBCPP_CONSTEXPR operator==(const Rep& r) const {return data_ == r.data_;}

    Rep& operator*=(Rep x) {data_ *= x.data_; return *this;}
    Rep& operator/=(Rep x) {data_ /= x.data_; return *this;}
};

#endif  // REP_H
