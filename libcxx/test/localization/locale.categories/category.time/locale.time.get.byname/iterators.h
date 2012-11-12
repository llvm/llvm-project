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

#endif  // ITERATORS_H
