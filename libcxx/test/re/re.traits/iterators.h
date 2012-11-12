#ifndef ITERATORS_H
#define ITERATORS_H

#include <iterator>

template <class It>
class input_iterator
{
    It it_;

    template <class U> friend class input_iterator;
public:
    typedef          std::input_iterator_tag                   iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    It base() const {return it_;}

    input_iterator() : it_() {}
    explicit input_iterator(It it) : it_(it) {}
    template <class U>
        input_iterator(const input_iterator<U>& u) :it_(u.it_) {}

    reference operator*() const {return *it_;}
    pointer operator->() const {return it_;}

    input_iterator& operator++() {++it_; return *this;}
    input_iterator operator++(int)
        {input_iterator tmp(*this); ++(*this); return tmp;}

    friend bool operator==(const input_iterator& x, const input_iterator& y)
        {return x.it_ == y.it_;}
    friend bool operator!=(const input_iterator& x, const input_iterator& y)
        {return !(x == y);}
};

template <class T, class U>
inline
bool
operator==(const input_iterator<T>& x, const input_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline
bool
operator!=(const input_iterator<T>& x, const input_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class forward_iterator
{
    It it_;

    template <class U> friend class forward_iterator;
public:
    typedef          std::forward_iterator_tag                 iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    It base() const {return it_;}

    forward_iterator() : it_() {}
    explicit forward_iterator(It it) : it_(it) {}
    template <class U>
        forward_iterator(const forward_iterator<U>& u) :it_(u.it_) {}

    reference operator*() const {return *it_;}
    pointer operator->() const {return it_;}

    forward_iterator& operator++() {++it_; return *this;}
    forward_iterator operator++(int)
        {forward_iterator tmp(*this); ++(*this); return tmp;}

    friend bool operator==(const forward_iterator& x, const forward_iterator& y)
        {return x.it_ == y.it_;}
    friend bool operator!=(const forward_iterator& x, const forward_iterator& y)
        {return !(x == y);}
};

template <class T, class U>
inline
bool
operator==(const forward_iterator<T>& x, const forward_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline
bool
operator!=(const forward_iterator<T>& x, const forward_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class bidirectional_iterator
{
    It it_;

    template <class U> friend class bidirectional_iterator;
public:
    typedef          std::bidirectional_iterator_tag           iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    It base() const {return it_;}

    bidirectional_iterator() : it_() {}
    explicit bidirectional_iterator(It it) : it_(it) {}
    template <class U>
        bidirectional_iterator(const bidirectional_iterator<U>& u) :it_(u.it_) {}

    reference operator*() const {return *it_;}
    pointer operator->() const {return it_;}

    bidirectional_iterator& operator++() {++it_; return *this;}
    bidirectional_iterator operator++(int)
        {bidirectional_iterator tmp(*this); ++(*this); return tmp;}

    bidirectional_iterator& operator--() {--it_; return *this;}
    bidirectional_iterator operator--(int)
        {bidirectional_iterator tmp(*this); --(*this); return tmp;}
};

template <class T, class U>
inline
bool
operator==(const bidirectional_iterator<T>& x, const bidirectional_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline
bool
operator!=(const bidirectional_iterator<T>& x, const bidirectional_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class random_access_iterator
{
    It it_;

    template <class U> friend class random_access_iterator;
public:
    typedef          std::random_access_iterator_tag           iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    It base() const {return it_;}

    random_access_iterator() : it_() {}
    explicit random_access_iterator(It it) : it_(it) {}
   template <class U>
        random_access_iterator(const random_access_iterator<U>& u) :it_(u.it_) {}

    reference operator*() const {return *it_;}
    pointer operator->() const {return it_;}

    random_access_iterator& operator++() {++it_; return *this;}
    random_access_iterator operator++(int)
        {random_access_iterator tmp(*this); ++(*this); return tmp;}

    random_access_iterator& operator--() {--it_; return *this;}
    random_access_iterator operator--(int)
        {random_access_iterator tmp(*this); --(*this); return tmp;}

    random_access_iterator& operator+=(difference_type n) {it_ += n; return *this;}
    random_access_iterator operator+(difference_type n) const
        {random_access_iterator tmp(*this); tmp += n; return tmp;}
    friend random_access_iterator operator+(difference_type n, random_access_iterator x)
        {x += n; return x;}
    random_access_iterator& operator-=(difference_type n) {return *this += -n;}
    random_access_iterator operator-(difference_type n) const
        {random_access_iterator tmp(*this); tmp -= n; return tmp;}

    reference operator[](difference_type n) const {return it_[n];}
};

template <class T, class U>
inline
bool
operator==(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
inline
bool
operator!=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(x == y);
}

template <class T, class U>
inline
bool
operator<(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() < y.base();
}

template <class T, class U>
inline
bool
operator<=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(y < x);
}

template <class T, class U>
inline
bool
operator>(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return y < x;
}

template <class T, class U>
inline
bool
operator>=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(x < y);
}

template <class T, class U>
inline
typename std::iterator_traits<T>::difference_type
operator-(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() - y.base();
}

#endif  // ITERATORS_H
