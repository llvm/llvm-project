#ifndef ITERATORS_H
#define ITERATORS_H

#include <iterator>

template <class It>
class output_iterator
{
    It it_;

    template <class U> friend class output_iterator;
public:
    typedef          std::output_iterator_tag                  iterator_category;
    typedef typename std::iterator_traits<It>::value_type      value_type;
    typedef typename std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename std::iterator_traits<It>::reference       reference;

    It base() const {return it_;}

    output_iterator() : it_() {}
    explicit output_iterator(It it) : it_(it) {}
    template <class U>
        output_iterator(const output_iterator<U>& u) :it_(u.it_) {}

    reference operator*() const {return *it_;}

    output_iterator& operator++() {++it_; return *this;}
    output_iterator operator++(int)
        {output_iterator tmp(*this); ++(*this); return tmp;}
};

#endif  // ITERATORS_H
