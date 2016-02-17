//===-- lld/Core/range.h - Iterator ranges ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Iterator range type based on c++1y range proposal.
///
/// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3350.html
///
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_RANGE_H
#define LLD_CORE_RANGE_H

#include "llvm/Support/Compiler.h"
#include <array>
#include <cassert>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace lld {
// Nothing in this namespace is part of the exported interface.
namespace detail {

using std::begin;
using std::end;

/// Used as the result type of undefined functions.
struct undefined {};

template <typename R> class begin_result {
  template <typename T> static auto check(T &&t) -> decltype(begin(t));
  static undefined check(...);
public:
  typedef decltype(check(std::declval<R>())) type;
};

template <typename R> class end_result {
  template <typename T> static auto check(T &&t) -> decltype(end(t));
  static undefined check(...);
public:
  typedef decltype(check(std::declval<R>())) type;
};

// Things that begin and end work on, in compatible ways, are
// ranges. [stmt.ranged]
template <typename R>
struct is_range : std::is_same<typename detail::begin_result<R>::type,
                               typename detail::end_result<R>::type> {};

// This currently requires specialization and doesn't work for
// detecting \c range<>s or iterators.  We should add
// \c contiguous_iterator_tag to fix that.
template <typename R> struct is_contiguous_range : std::false_type {};
template <typename R>
struct is_contiguous_range<R &> : is_contiguous_range<R> {};
template <typename R>
struct is_contiguous_range <R &&> : is_contiguous_range<R> {};
template <typename R>
struct is_contiguous_range<const R> : is_contiguous_range<R> {};

template <typename T, size_t N>
struct is_contiguous_range<T[N]> : std::true_type {};
template <typename T, size_t N>
struct is_contiguous_range<const T[N]> : std::true_type {};
template <typename T, size_t N>
struct is_contiguous_range<std::array<T, N> > : std::true_type {};
template <typename charT, typename traits, typename Allocator>
struct is_contiguous_range<
    std::basic_string<charT, traits, Allocator> > : std::true_type {};
template <typename T, typename Allocator>
struct is_contiguous_range<std::vector<T, Allocator> > : std::true_type {};

// Removes cv qualifiers from all levels of a multi-level pointer
// type, not just the type level.
template <typename T> struct remove_all_cv_ptr {
  typedef T type;
};
template <typename T> struct remove_all_cv_ptr<T *> {
  typedef typename remove_all_cv_ptr<T>::type *type;
};
template <typename T> struct remove_all_cv_ptr<const T> {
  typedef typename remove_all_cv_ptr<T>::type type;
};
template <typename T> struct remove_all_cv_ptr<volatile T> {
  typedef typename remove_all_cv_ptr<T>::type type;
};
template <typename T> struct remove_all_cv_ptr<const volatile T> {
  typedef typename remove_all_cv_ptr<T>::type type;
};

template <typename From, typename To>
struct conversion_preserves_array_indexing : std::false_type {};

template <typename FromVal, typename ToVal>
struct conversion_preserves_array_indexing<FromVal *,
                                           ToVal *> : std::integral_constant<
    bool, std::is_convertible<FromVal *, ToVal *>::value &&
    std::is_same<typename remove_all_cv_ptr<FromVal>::type,
                 typename remove_all_cv_ptr<ToVal>::type>::value> {};

template <typename T>
LLVM_CONSTEXPR auto adl_begin(T &&t) -> decltype(begin(t)) {
  return begin(std::forward<T>(t));
}

template <typename T> LLVM_CONSTEXPR auto adl_end(T &&t) -> decltype(end(t)) {
  return end(std::forward<T>(t));
}
} // end namespace detail

/// A \c std::range<Iterator> represents a half-open iterator range
/// built from two iterators, \c 'begin', and \c 'end'.  If \c end is
/// not reachable from \c begin, the behavior is undefined.
///
/// The mutability of elements of the range is controlled by the
/// Iterator argument.  Instantiate
/// <code>range<<var>Foo</var>::iterator></code> or
/// <code>range<<var>T</var>*></code>, or call
/// <code>make_range(<var>non_const_container</var>)</code>, and you
/// get a mutable range.  Instantiate
/// <code>range<<var>Foo</var>::const_iterator></code> or
/// <code>range<const <var>T</var>*></code>, or call
/// <code>make_range(<var>const_container</var>)</code>, and you get a
/// constant range.
///
/// \todo Inherit from std::pair<Iterator, Iterator>?
///
/// \todo This interface contains some functions that could be
/// provided as free algorithms rather than member functions, and all
/// of the <code>pop_*()</code> functions could be replaced by \c
/// slice() at the cost of some extra iterator copies.  This makes
/// them more awkward to use, but makes it easier for users to write
/// their own types that follow the same interface. On the other hand,
/// a \c range_facade could be provided to help users write new
/// ranges, and it could provide the members.  Such functions are
/// marked with a note in their documentation.  (Of course, all of
/// these member functions could be provided as free functions using
/// the iterator access methods, but one goal here is to allow people
/// to program without touching iterators at all.)
template <typename Iterator> class range {
  Iterator begin_, end_;
public:
  /// \name types
  /// @{

  /// The iterator category of \c Iterator.
  /// \todo Consider defining range categories. If they don't add
  /// anything over the corresponding iterator categories, then
  /// they're probably not worth defining.
  typedef typename std::iterator_traits<
      Iterator>::iterator_category iterator_category;
  /// The type of elements of the range. Not cv-qualified.
  typedef typename std::iterator_traits<Iterator>::value_type value_type;
  /// The type of the size of the range and offsets within the range.
  typedef typename std::iterator_traits<
      Iterator>::difference_type difference_type;
  /// The return type of element access methods: \c front(), \c back(), etc.
  typedef typename std::iterator_traits<Iterator>::reference reference;
  typedef typename std::iterator_traits<Iterator>::pointer pointer;
  /// @}

  /// \name constructors
  /// @{

  /// Creates a range of default-constructed (<em>not</em>
  /// value-initialized) iterators.  For most \c Iterator types, this
  /// will be an invalid range.
  range() : begin_(), end_() {}

  /// \pre \c end is reachable from \c begin.
  /// \post <code>this->begin() == begin && this->end() == end</code>
  LLVM_CONSTEXPR range(Iterator begin, Iterator end)
      : begin_(begin), end_(end) {}

  /// \par Participates in overload resolution if:
  ///   - \c Iterator is not a pointer type,
  ///   - \c begin(r) and \c end(r) return the same type, and
  ///   - that type is convertible to \c Iterator.
  ///
  /// \todo std::begin and std::end are overloaded between T& and
  /// const T&, which means that if a container has only a non-const
  /// begin or end method, then it's ill-formed to pass an rvalue to
  /// the free function.  To avoid that problem, we don't use
  /// std::forward<> here, so begin() and end() are always called with
  /// an lvalue.  Another option would be to insist that rvalue
  /// arguments to range() must have const begin() and end() methods.
  template <typename R> LLVM_CONSTEXPR range(
      R &&r,
      typename std::enable_if<
        !std::is_pointer<Iterator>::value &&
        detail::is_range<R>::value &&
        std::is_convertible<typename detail::begin_result<R>::type,
                            Iterator>::value>::type* = nullptr)
      : begin_(detail::adl_begin(r)), end_(detail::adl_end(r)) {}

  /// This constructor creates a \c range<T*> from any range with
  /// contiguous iterators. Because dereferencing a past-the-end
  /// iterator can be undefined behavior, empty ranges get initialized
  /// with \c nullptr rather than \c &*begin().
  ///
  /// \par Participates in overload resolution if:
  ///   - \c Iterator is a pointer type \c T*,
  ///   - \c begin(r) and \c end(r) return the same type,
  ///   - elements \c i of that type satisfy the invariant
  ///     <code>&*(i + N) == (&*i) + N</code>, and
  ///   - The result of <code>&*begin()</code> is convertible to \c T*
  ///     using only qualification conversions [conv.qual] (since
  ///     pointer conversions stop the pointer from pointing to an
  ///     array element).
  ///
  /// \todo The <code>&*(i + N) == (&*i) + N</code> invariant is
  /// currently impossible to check for user-defined types.  We need a
  /// \c contiguous_iterator_tag to let users assert it.
  template <typename R> LLVM_CONSTEXPR range(
      R &&r,
      typename std::enable_if<
        std::is_pointer<Iterator>::value &&
        detail::is_contiguous_range<R>::value
      // MSVC returns false for this in this context, but not if we lift it out of the
      // constructor.
#ifndef _MSC_VER
        && detail::conversion_preserves_array_indexing<
             decltype(&*detail::adl_begin(r)), Iterator>::value
#endif
      >::type* = nullptr)
      : begin_((detail::adl_begin(r) == detail::adl_end(r) &&
                !std::is_pointer<decltype(detail::adl_begin(r))>::value)
               // For non-pointers, &*begin(r) is only defined behavior
               // if there's an element there.  Otherwise, use nullptr
               // since the user can't dereference it anyway.  This _is_
               // detectable.
               ? nullptr : &*detail::adl_begin(r)),
        end_(begin_ + (detail::adl_end(r) - detail::adl_begin(r))) {}

  /// @}

  /// \name iterator access
  /// @{
  LLVM_CONSTEXPR Iterator begin() const { return begin_; }
  LLVM_CONSTEXPR Iterator end() const { return end_; }
  /// @}

  /// \name element access
  /// @{

  /// \par Complexity:
  /// O(1)
  /// \pre \c !empty()
  /// \returns a reference to the element at the front of the range.
  LLVM_CONSTEXPR reference front() const { return *begin(); }

  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to \c
  /// std::bidirectional_iterator_tag.
  ///
  /// \par Complexity:
  /// O(2) (Involves copying and decrementing an iterator, so not
  /// quite as cheap as \c front())
  ///
  /// \pre \c !empty()
  /// \returns a reference to the element at the front of the range.
  LLVM_CONSTEXPR reference back() const {
    static_assert(
        std::is_convertible<iterator_category,
                            std::bidirectional_iterator_tag>::value,
        "Can only retrieve the last element of a bidirectional range.");
    using std::prev;
    return *prev(end());
  }

  /// This method is drawn from scripting language indexing.  It
  /// indexes std::forward from the beginning of the range if the argument
  /// is positive, or backwards from the end of the array if the
  /// argument is negative.
  ///
  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to \c
  /// std::random_access_iterator_tag.
  ///
  /// \par Complexity:
  /// O(1)
  ///
  /// \pre <code>abs(index) < size() || index == -size()</code>
  ///
  /// \returns if <code>index >= 0</code>, a reference to the
  /// <code>index</code>'th element in the range. Otherwise, a
  /// reference to the <code>size()+index</code>'th element.
  LLVM_CONSTEXPR reference operator[](difference_type index) const {
    static_assert(std::is_convertible<iterator_category,
                                      std::random_access_iterator_tag>::value,
                  "Can only index into a random-access range.");
    // Less readable construction for constexpr support.
    return index < 0 ? end()[index]
                     : begin()[index];
  }
  /// @}

  /// \name size
  /// @{

  /// \par Complexity:
  /// O(1)
  /// \returns \c true if the range contains no elements.
  LLVM_CONSTEXPR bool empty() const { return begin() == end(); }

  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to
  /// \c std::forward_iterator_tag.
  ///
  /// \par Complexity:
  /// O(1) if \c iterator_category is convertible to \c
  /// std::random_access_iterator_tag. O(<code>size()</code>)
  /// otherwise.
  ///
  /// \returns the number of times \c pop_front() can be called before
  /// \c empty() becomes true.
  LLVM_CONSTEXPR difference_type size() const {
    static_assert(std::is_convertible<iterator_category,
                                      std::forward_iterator_tag>::value,
                  "Calling size on an input range would destroy the range.");
    return dispatch_size(iterator_category());
  }
  /// @}

  /// \name traversal from the beginning of the range
  /// @{

  /// Advances the beginning of the range by one element.
  /// \pre \c !empty()
  void pop_front() { ++begin_; }

  /// Advances the beginning of the range by \c n elements.
  ///
  /// \par Complexity:
  /// O(1) if \c iterator_category is convertible to \c
  /// std::random_access_iterator_tag, O(<code>n</code>) otherwise.
  ///
  /// \pre <code>n >= 0</code>, and there must be at least \c n
  /// elements in the range.
  void pop_front(difference_type n) { advance(begin_, n); }

  /// Advances the beginning of the range by at most \c n elements,
  /// stopping if the range becomes empty.  A negative argument causes
  /// no change.
  ///
  /// \par Complexity:
  /// O(1) if \c iterator_category is convertible to \c
  /// std::random_access_iterator_tag, O(<code>min(n,
  /// <var>#-elements-in-range</var>)</code>) otherwise.
  ///
  /// \note Could be provided as a free function with little-to-no
  /// loss in efficiency.
  void pop_front_upto(difference_type n) {
    advance_upto(begin_, std::max<difference_type>(0, n), end_,
                 iterator_category());
  }

  /// @}

  /// \name traversal from the end of the range
  /// @{

  /// Moves the end of the range earlier by one element.
  ///
  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to
  /// \c std::bidirectional_iterator_tag.
  ///
  /// \par Complexity:
  /// O(1)
  ///
  /// \pre \c !empty()
  void pop_back() {
    static_assert(std::is_convertible<iterator_category,
                                      std::bidirectional_iterator_tag>::value,
                  "Can only access the end of a bidirectional range.");
    --end_;
  }

  /// Moves the end of the range earlier by \c n elements.
  ///
  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to
  /// \c std::bidirectional_iterator_tag.
  ///
  /// \par Complexity:
  /// O(1) if \c iterator_category is convertible to \c
  /// std::random_access_iterator_tag, O(<code>n</code>) otherwise.
  ///
  /// \pre <code>n >= 0</code>, and there must be at least \c n
  /// elements in the range.
  void pop_back(difference_type n) {
    static_assert(std::is_convertible<iterator_category,
                                      std::bidirectional_iterator_tag>::value,
                  "Can only access the end of a bidirectional range.");
    advance(end_, -n);
  }

  /// Moves the end of the range earlier by <code>min(n,
  /// size())</code> elements.  A negative argument causes no change.
  ///
  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to
  /// \c std::bidirectional_iterator_tag.
  ///
  /// \par Complexity:
  /// O(1) if \c iterator_category is convertible to \c
  /// std::random_access_iterator_tag, O(<code>min(n,
  /// <var>#-elements-in-range</var>)</code>) otherwise.
  ///
  /// \note Could be provided as a free function with little-to-no
  /// loss in efficiency.
  void pop_back_upto(difference_type n) {
    static_assert(std::is_convertible<iterator_category,
                                      std::bidirectional_iterator_tag>::value,
                  "Can only access the end of a bidirectional range.");
    advance_upto(end_, -std::max<difference_type>(0, n), begin_,
                 iterator_category());
  }

  /// @}

  /// \name creating derived ranges
  /// @{

  /// Divides the range into two pieces at \c index, where a positive
  /// \c index represents an offset from the beginning of the range
  /// and a negative \c index represents an offset from the end.
  /// <code>range[index]</code> is the first element in the second
  /// piece.  If <code>index >= size()</code>, the second piece
  /// will be empty. If <code>index < -size()</code>, the first
  /// piece will be empty.
  ///
  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to
  /// \c std::forward_iterator_tag.
  ///
  /// \par Complexity:
  ///   - If \c iterator_category is convertible to \c
  ///     std::random_access_iterator_tag: O(1)
  ///   - Otherwise, if \c iterator_category is convertible to \c
  ///     std::bidirectional_iterator_tag, \c abs(index) iterator increments
  ///     or decrements
  ///   - Otherwise, if <code>index >= 0</code>,  \c index iterator
  ///     increments
  ///   - Otherwise, <code>size() + (size() + index)</code>
  ///     iterator increments.
  ///
  /// \returns a pair of adjacent ranges.
  ///
  /// \post
  ///   - <code>result.first.size() == min(index, this->size())</code>
  ///   - <code>result.first.end() == result.second.begin()</code>
  ///   - <code>result.first.size() + result.second.size()</code> <code>==
  ///     this->size()</code>
  ///
  /// \todo split() could take an arbitrary number of indices and
  /// return an <code>N+1</code>-element \c tuple<>. This is tricky to
  /// implement with negative indices in the optimal number of
  /// increments or decrements for a bidirectional iterator, but it
  /// should be possible.  Do we want it?
  std::pair<range, range> split(difference_type index) const {
    static_assert(
        std::is_convertible<iterator_category,
                            std::forward_iterator_tag>::value,
        "Calling split on a non-std::forward range would return a useless "
        "first result.");
    if (index >= 0) {
      range second = *this;
      second.pop_front_upto(index);
      return make_pair(range(begin(), second.begin()), second);
    } else {
      return dispatch_split_neg(index, iterator_category());
    }
  }

  /// \returns A sub-range from \c start to \c stop (not including \c
  /// stop, as usual).  \c start and \c stop are interpreted as for
  /// <code>operator[]</code>, with negative values offsetting from
  /// the end of the range.  Omitting the \c stop argument makes the
  /// sub-range continue to the end of the original range. Positive
  /// arguments saturate to the end of the range, and negative
  /// arguments saturate to the beginning.  If \c stop is before \c
  /// start, returns an empty range beginning and ending at \c start.
  ///
  /// \par Ill-formed unless:
  /// \c iterator_category is convertible to
  /// \c std::forward_iterator_tag.
  ///
  /// \par Complexity:
  ///   - If \c iterator_category is convertible to \c
  ///     std::random_access_iterator_tag: O(1)
  ///   - Otherwise, if \c iterator_category is convertible to \c
  ///     std::bidirectional_iterator_tag, at most <code>min(abs(start),
  ///     size()) + min(abs(stop), size())</code> iterator
  ///     increments or decrements
  ///   - Otherwise, if <code>start >= 0 && stop >= 0</code>,
  ///     <code>max(start, stop)</code> iterator increments
  ///   - Otherwise, <code>size() + max(start', stop')</code>
  ///     iterator increments, where \c start' and \c stop' are the
  ///     offsets of the elements \c start and \c stop refer to.
  ///
  /// \note \c slice(start) should be implemented with a different
  /// overload, rather than defaulting \c stop to
  /// <code>numeric_limits<difference_type>::max()</code>, because
  /// using a default would force non-random-access ranges to use an
  /// O(<code>size()</code>) algorithm to compute the end rather
  /// than the O(1) they're capable of.
  range slice(difference_type start, difference_type stop) const {
    static_assert(
        std::is_convertible<iterator_category,
                            std::forward_iterator_tag>::value,
        "Calling slice on a non-std::forward range would destroy the original "
        "range.");
    return dispatch_slice(start, stop, iterator_category());
  }

  range slice(difference_type start) const {
    static_assert(
        std::is_convertible<iterator_category,
                            std::forward_iterator_tag>::value,
        "Calling slice on a non-std::forward range would destroy the original "
        "range.");
    return split(start).second;
  }

  /// @}

private:
  // advance_upto: should be added to <algorithm>, but I'll use it as
  // a helper function here.
  //
  // These return the number of increments that weren't applied
  // because we ran into 'limit' (or 0 if we didn't run into limit).
  static difference_type advance_upto(Iterator &it, difference_type n,
                                      Iterator limit, std::input_iterator_tag) {
    if (n < 0)
      return 0;
    while (it != limit && n > 0) {
      ++it;
      --n;
    }
    return n;
  }

  static difference_type advance_upto(Iterator &it, difference_type n,
                                      Iterator limit,
                                      std::bidirectional_iterator_tag) {
    if (n < 0) {
      while (it != limit && n < 0) {
        --it;
        ++n;
      }
    } else {
      while (it != limit && n > 0) {
        ++it;
        --n;
      }
    }
    return n;
  }

  static difference_type advance_upto(Iterator &it, difference_type n,
                                      Iterator limit,
                                      std::random_access_iterator_tag) {
    difference_type distance = limit - it;
    if (distance < 0)
      assert(n <= 0);
    else if (distance > 0)
      assert(n >= 0);

    if (abs(distance) > abs(n)) {
      it += n;
      return 0;
    } else {
      it = limit;
      return n - distance;
    }
  }

  // Dispatch functions.
  difference_type dispatch_size(std::forward_iterator_tag) const {
    return std::distance(begin(), end());
  }

  LLVM_CONSTEXPR difference_type dispatch_size(
      std::random_access_iterator_tag) const {
    return end() - begin();
  }

  std::pair<range, range> dispatch_split_neg(difference_type index,
                                             std::forward_iterator_tag) const {
    assert(index < 0);
    difference_type size = this->size();
    return split(std::max<difference_type>(0, size + index));
  }

  std::pair<range, range> dispatch_split_neg(
      difference_type index, std::bidirectional_iterator_tag) const {
    assert(index < 0);
    range first = *this;
    first.pop_back_upto(-index);
    return make_pair(first, range(first.end(), end()));
  }

  range dispatch_slice(difference_type start, difference_type stop,
                       std::forward_iterator_tag) const {
    if (start < 0 || stop < 0) {
      difference_type size = this->size();
      if (start < 0)
        start = std::max<difference_type>(0, size + start);
      if (stop < 0)
        stop = size + stop; // Possibly negative; will be fixed in 2 lines.
    }
    stop = std::max<difference_type>(start, stop);

    Iterator first = begin();
    advance_upto(first, start, end(), iterator_category());
    Iterator last = first;
    advance_upto(last, stop - start, end(), iterator_category());
    return range(first, last);
  }

  range dispatch_slice(const difference_type start, const difference_type stop,
                       std::bidirectional_iterator_tag) const {
    Iterator first;
    if (start < 0) {
      first = end();
      advance_upto(first, start, begin(), iterator_category());
    } else {
      first = begin();
      advance_upto(first, start, end(), iterator_category());
    }
    Iterator last;
    if (stop < 0) {
      last = end();
      advance_upto(last, stop, first, iterator_category());
    } else {
      if (start >= 0) {
        last = first;
        if (stop > start)
          advance_upto(last, stop - start, end(), iterator_category());
      } else {
        // Complicated: 'start' walked from the end of the sequence,
        // but 'stop' needs to walk from the beginning.
        Iterator dummy = begin();
        // Walk up to 'stop' increments from begin(), stopping when we
        // get to 'first', and capturing the remaining number of
        // increments.
        difference_type increments_past_start =
            advance_upto(dummy, stop, first, iterator_category());
        if (increments_past_start == 0) {
          // If this is 0, then stop was before start.
          last = first;
        } else {
          // Otherwise, count that many spaces beyond first.
          last = first;
          advance_upto(last, increments_past_start, end(), iterator_category());
        }
      }
    }
    return range(first, last);
  }

  range dispatch_slice(difference_type start, difference_type stop,
                       std::random_access_iterator_tag) const {
    const difference_type size = this->size();
    if (start < 0)
      start = size + start;
    if (start < 0)
      start = 0;
    if (start > size)
      start = size;

    if (stop < 0)
      stop = size + stop;
    if (stop < start)
      stop = start;
    if (stop > size)
      stop = size;

    return range(begin() + start, begin() + stop);
  }
};

/// \name deducing constructor wrappers
/// \relates std::range
/// \xmlonly <nonmember/> \endxmlonly
///
/// These functions do the same thing as the constructor with the same
/// signature. They just allow users to avoid writing the iterator
/// type.
/// @{

/// \todo I'd like to define a \c make_range taking a single iterator
/// argument representing the beginning of a range that ends with a
/// default-constructed \c Iterator.  This would help with using
/// iterators like \c istream_iterator.  However, using just \c
/// make_range() could be confusing and lead to people writing
/// incorrect ranges of more common iterators. Is there a better name?
template <typename Iterator>
LLVM_CONSTEXPR range<Iterator> make_range(Iterator begin, Iterator end) {
  return range<Iterator>(begin, end);
}

/// \par Participates in overload resolution if:
/// \c begin(r) and \c end(r) return the same type.
template <typename Range> LLVM_CONSTEXPR auto make_range(
    Range &&r,
    typename std::enable_if<detail::is_range<Range>::value>::type* = nullptr)
    -> range<decltype(detail::adl_begin(r))> {
  return range<decltype(detail::adl_begin(r))>(r);
}

/// \par Participates in overload resolution if:
///   - \c begin(r) and \c end(r) return the same type,
///   -  that type satisfies the invariant that <code>&*(i + N) ==
///      (&*i) + N</code>, and
///   - \c &*begin(r) has a pointer type.
template <typename Range> LLVM_CONSTEXPR auto make_ptr_range(
    Range &&r,
    typename std::enable_if<
      detail::is_contiguous_range<Range>::value &&
      std::is_pointer<decltype(&*detail::adl_begin(r))>::value>::type* =
      nullptr) -> range<decltype(&*detail::adl_begin(r))> {
  return range<decltype(&*detail::adl_begin(r))>(r);
}
/// @}
} // end namespace lld

#endif // LLD_CORE_RANGE_H
