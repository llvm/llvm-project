// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FORMAT_BUFFER_H
#define _LIBCPP___FORMAT_BUFFER_H

#include <__algorithm/copy_n.h>
#include <__algorithm/fill_n.h>
#include <__algorithm/max.h>
#include <__algorithm/min.h>
#include <__algorithm/ranges_copy.h>
#include <__algorithm/ranges_copy_n.h>
#include <__algorithm/transform.h>
#include <__algorithm/unwrap_iter.h>
#include <__concepts/same_as.h>
#include <__config>
#include <__format/concepts.h>
#include <__format/enable_insertable.h>
#include <__format/format_to_n_result.h>
#include <__iterator/back_insert_iterator.h>
#include <__iterator/concepts.h>
#include <__iterator/incrementable_traits.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/wrap_iter.h>
#include <__memory/addressof.h>
#include <__memory/allocate_at_least.h>
#include <__memory/allocator.h>
#include <__memory/allocator_traits.h>
#include <__memory/construct_at.h>
#include <__memory/ranges_construct_at.h>
#include <__memory/uninitialized_algorithms.h>
#include <__type_traits/add_pointer.h>
#include <__type_traits/conditional.h>
#include <__utility/exception_guard.h>
#include <__utility/move.h>
#include <climits> // LLVM-20 remove
#include <cstddef>
#include <stdexcept>
#include <string_view>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

namespace __format {

// A helper to limit the total size of code units written.
class _LIBCPP_HIDE_FROM_ABI __max_output_size {
public:
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI explicit __max_output_size(size_t __max_size) : __max_size_{__max_size} {}

  // This function adjusts the size of a (bulk) write operations. It ensures the
  // number of code units written by a __output_buffer never exceed
  // __max_size_ code units.
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t __write_request(size_t __code_units) {
    size_t __result =
        __code_units_written_ < __max_size_ ? std::min(__code_units, __max_size_ - __code_units_written_) : 0;
    __code_units_written_ += __code_units;
    return __result;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t __code_units_written() const noexcept { return __code_units_written_; }

private:
  size_t __max_size_;
  // The code units that would have been written if there was no limit.
  // format_to_n returns this value.
  size_t __code_units_written_{0};
};

/// A "buffer" that handles writing to the proper iterator.
///
/// This helper is used together with the @ref back_insert_iterator to offer
/// type-erasure for the formatting functions. This reduces the number to
/// template instantiations.
///
/// The design of the class is being changed to improve performance and do some
/// code cleanups.
/// The original design (as shipped up to LLVM-19) uses the following design:
/// - There is an external object that connects the buffer to the output.
/// - The class constructor stores a function pointer to a grow function and a
///   type-erased pointer to the object that does the grow.
/// - When writing data to the buffer would exceed the external buffer's
///   capacity it requests the external buffer to flush its contents.
///
/// The new design tries to solve some issues with the current design:
/// - The buffer used is a fixed-size buffer, benchmarking shows that using a
///   dynamic allocated buffer has performance benefits.
/// - Implementing P3107R5 "Permit an efficient implementation of std::print"
///   is not trivial with the current buffers. Using the code from this series
///   makes it trivial.
///
/// This class is ABI-tagged, still the new design does not change the size of
/// objects of this class.
///
/// The new design is the following.
/// - There is an external object that connects the buffer to the output.
/// - This buffer object:
///   - inherits publicly from this class.
///   - has a static or dynamic buffer.
///   - has a static member function to make space in its buffer write
///     operations. This can be done by increasing the size of the internal
///     buffer or by writing the contents of the buffer to the output iterator.
///
///     This member function is a constructor argument, so its name is not
///     fixed. The code uses the name __prepare_write.
/// - The number of output code units can be limited by a __max_output_size
///   object. This is used in format_to_n This object:
///   - Contains the maximum number of code units to be written.
///   - Contains the number of code units that are requested to be written.
///     This number is returned to the user of format_to_n.
///   - The write functions call objects __request_write member function.
///     This function:
///     - Updates the number of code units that are requested to be written.
///     - Returns the number of code units that can be written without
///       exceeding the maximum number of code units to be written.
///
/// Documentation for the buffer usage members:
/// - __ptr_ the start of the buffer.
/// - __capacity_ the number of code units that can be written.
///   This means [__ptr_, __ptr_ + __capacity_) is a valid range to write to.
/// - __size_ the number of code units written in the buffer. The next code
///   unit will be written at __ptr_ + __size_. This __size_ may NOT contain
///   the total number of code units written by the __output_buffer. Whether or
///   not it does depends on the sub-class used. Typically the total number of
///   code units written is not interesting. It is interesting for format_to_n
///   which has its own way to track this number.
///
/// Documentation for the buffer changes function:
/// The subclasses have a function with the following signature:
///
///   static void __prepare_write(
///     __output_buffer<_CharT>& __buffer, size_t __code_units);
///
/// This function is called when a write function writes more code units than
/// the buffer' available space. When an __max_output_size object is provided
/// the number of code units is the number of code units returned from
/// __max_output_size::__request_write function.
///
/// - The __buffer contains *this. Since the class containing this function
///   inherits from __output_buffer it's save to cast it to the subclass being
///   used.
/// - The __code_units is the number of code units the caller will write + 1.
///   - This value does not take the avaiable space of the buffer into account.
///   - The push_back function is more efficient when writing before resizing,
///     this means the buffer should always have room for one code unit. Hence
///     the + 1 is the size.
/// - When the function returns there is room for at least one code unit. There
///   is no requirement there is room for __code_units code units:
///   - The class has some "bulk" operations. For example, __copy which copies
///     the contents of a basic_string_view to the output. If the sub-class has
///     a fixed size buffer the size of the basic_string_view may be larger
///     than the buffer. In that case it's impossible to honor the requested
///     size.
///   - The at least one code unit makes sure the entire output can be written.
///     (Obviously making room one code unit at a time is slow and
///     it's recommended to return a larger available space.)
///   - When the buffer has room for at least one code unit the function may be
///     a no-op.
/// - When the function makes space for more code units it uses one for these
///   functions to signal the change:
///   - __buffer_flushed()
///     - This function is typically used for a fixed sized buffer.
///     - The current contents of [__ptr_, __ptr_ + __size_) have been
///       processed.
///     - __ptr_ remains unchanged.
///     - __capacity_ remains unchanged.
///     - __size_ will be set to 0.
///   - __buffer_moved(_CharT* __ptr, size_t __capacity)
///     - This function is typically used for a dynamic sized buffer. There the
///       location of the buffer changes due to reallocations.
///     - __ptr_ will be set to __ptr. (This value may be the old value of
///       __ptr_).
///     - __capacity_ will be set to __capacity. (This value may be the old
///       value of       __capacity_).
///     - __size_ remains unchanged,
///     - The range [__ptr, __ptr + __size_) contains the original data of the
///       range  [__ptr_, __ptr_ + __size_).
///
/// The push_back function expects a valid buffer and a capacity of at least 1.
/// This means:
/// - The class is constructed with a valid buffer,
/// - __buffer_moved is called with a valid buffer is used before the first
///   write operation,
/// - no write function is ever called, or
/// - the class is constructed with a __max_output_size object with __max_size 0.
///
/// The latter option allows formatted_size to use the output buffer without
/// ever writing anything to the buffer.
template <__fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __output_buffer {
public:
  using value_type           = _CharT;
  using __prepare_write_type = void (*)(__output_buffer<_CharT>&, size_t);

  template <class _Tp> // Deprecated LLVM-19 function.
  _LIBCPP_HIDE_FROM_ABI explicit __output_buffer(_CharT* __ptr, size_t __capacity, _Tp* __obj)
      : __ptr_(__ptr),
        __capacity_(__capacity),
        __flush_([](_CharT* __p, size_t __n, void* __o) { static_cast<_Tp*>(__o)->__flush(__p, __n); }),
        __data_{.__version_llvm_20__ = false, .__obj_ = reinterpret_cast<uintptr_t>(__obj) >> 1} {}

  // New LLVM-20 function.
  [[nodiscard]]
  _LIBCPP_HIDE_FROM_ABI explicit __output_buffer(_CharT* __ptr, size_t __capacity, __prepare_write_type __prepare_write)
      : __output_buffer{__ptr, __capacity, __prepare_write, nullptr} {}

  // New LLVM-20 function.
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI explicit __output_buffer(
      _CharT* __ptr, size_t __capacity, __prepare_write_type __prepare_write, __max_output_size* __max_output_size)
      : __ptr_(__ptr),
        __capacity_(__capacity),
        __prepare_write_(__prepare_write),
        __data_{.__version_llvm_20_ = true, .__max_output_size_ = reinterpret_cast<uintptr_t>(__max_output_size) >> 1} {
  }

  // Deprecated LLVM-19 function.
  _LIBCPP_HIDE_FROM_ABI void __reset(_CharT* __ptr, size_t __capacity) {
    __ptr_      = __ptr;
    __capacity_ = __capacity;
  }

  // New LLVM-20 function.
  _LIBCPP_HIDE_FROM_ABI void __buffer_flused() { __size_ = 0; }

  // New LLVM-20 function.
  _LIBCPP_HIDE_FROM_ABI void __buffer_moved(_CharT* __ptr, size_t __capacity) {
    __ptr_      = __ptr;
    __capacity_ = __capacity;
  }

  _LIBCPP_HIDE_FROM_ABI auto __make_output_iterator() { return std::back_insert_iterator{*this}; }

  // Used in std::back_insert_iterator.
  _LIBCPP_HIDE_FROM_ABI void push_back(_CharT __c) {
    if (__data_.__version_llvm_20_ && __data_.__max_output_size_ &&
        reinterpret_cast<__max_output_size*>(__data_.__max_output_size_ << 1)->__write_request(1) == 0)
      return;

    __ptr_[__size_++] = __c;

    // Profiling showed flushing after adding is more efficient than flushing
    // when entering the function.
    if (__size_ == __capacity_)
      __flush(0);
  }

  /// Copies the input __str to the buffer.
  ///
  /// Since some of the input is generated by std::to_chars, there needs to be a
  /// conversion when _CharT is wchar_t.
  template <__fmt_char_type _InCharT>
  _LIBCPP_HIDE_FROM_ABI void __copy(basic_string_view<_InCharT> __str) {
    // When the underlying iterator is a simple iterator the __capacity_ is
    // infinite. For a string or container back_inserter it isn't. This means
    // that adding a large string to the buffer can cause some overhead. In that
    // case a better approach could be:
    // - flush the buffer
    // - container.append(__str.begin(), __str.end());
    // The same holds true for the fill.
    // For transform it might be slightly harder, however the use case for
    // transform is slightly less common; it converts hexadecimal values to
    // upper case. For integral these strings are short.
    // TODO FMT Look at the improvements above.
    size_t __n = __str.size();
    if (__data_.__version_llvm_20_ && __data_.__max_output_size_) {
      __n = reinterpret_cast<__max_output_size*>(__data_.__max_output_size_ << 1)->__write_request(__n);
      if (__n == 0)
        return;
    }

    __flush_on_overflow(__n);
    if (__n < __capacity_) { // push_back requires the buffer to have room for at least one character (so use <).
      std::copy_n(__str.data(), __n, std::addressof(__ptr_[__size_]));
      __size_ += __n;
      return;
    }

    // The output doesn't fit in the internal buffer.
    // Copy the data in "__capacity_" sized chunks.
    _LIBCPP_ASSERT_INTERNAL(__size_ == 0, "the buffer should be flushed by __flush_on_overflow");
    const _InCharT* __first = __str.data();
    do {
      size_t __chunk = std::min(__n, __capacity_ - __size_);
      std::copy_n(__first, __chunk, std::addressof(__ptr_[__size_]));
      __size_ = __chunk;
      __first += __chunk;
      __n -= __chunk;
      __flush(__n);
    } while (__n);
  }

  /// A std::transform wrapper.
  ///
  /// Like @ref __copy it may need to do type conversion.
  template <contiguous_iterator _Iterator,
            class _UnaryOperation,
            __fmt_char_type _InCharT = typename iterator_traits<_Iterator>::value_type>
  _LIBCPP_HIDE_FROM_ABI void __transform(_Iterator __first, _Iterator __last, _UnaryOperation __operation) {
    _LIBCPP_ASSERT_INTERNAL(__first <= __last, "not a valid range");

    size_t __n = static_cast<size_t>(__last - __first);
    if (__data_.__version_llvm_20_ && __data_.__max_output_size_) {
      __n = reinterpret_cast<__max_output_size*>(__data_.__max_output_size_ << 1)->__write_request(__n);
      if (__n == 0)
        return;
    }

    __flush_on_overflow(__n);
    if (__n < __capacity_) { //  push_back requires the buffer to have room for at least one character (so use <).
      std::transform(__first, __last, std::addressof(__ptr_[__size_]), std::move(__operation));
      __size_ += __n;
      return;
    }

    // The output doesn't fit in the internal buffer.
    // Transform the data in "__capacity_" sized chunks.
    _LIBCPP_ASSERT_INTERNAL(__size_ == 0, "the buffer should be flushed by __flush_on_overflow");
    do {
      size_t __chunk = std::min(__n, __capacity_ - __size_);
      std::transform(__first, __first + __chunk, std::addressof(__ptr_[__size_]), __operation);
      __size_ = __chunk;
      __first += __chunk;
      __n -= __chunk;
      __flush(__n);
    } while (__n);
  }

  /// A \c fill_n wrapper.
  _LIBCPP_HIDE_FROM_ABI void __fill(size_t __n, _CharT __value) {
    if (__data_.__version_llvm_20_ && __data_.__max_output_size_) {
      __n = reinterpret_cast<__max_output_size*>(__data_.__max_output_size_ << 1)->__write_request(__n);
      if (__n == 0)
        return;
    }

    __flush_on_overflow(__n);
    if (__n < __capacity_) { //  push_back requires the buffer to have room for at least one character (so use <).
      std::fill_n(std::addressof(__ptr_[__size_]), __n, __value);
      __size_ += __n;
      return;
    }

    // The output doesn't fit in the internal buffer.
    // Fill the buffer in "__capacity_" sized chunks.
    _LIBCPP_ASSERT_INTERNAL(__size_ == 0, "the buffer should be flushed by __flush_on_overflow");
    do {
      size_t __chunk = std::min(__n, __capacity_);
      std::fill_n(std::addressof(__ptr_[__size_]), __chunk, __value);
      __size_ = __chunk;
      __n -= __chunk;
      __flush(__n);
    } while (__n);
  }

  _LIBCPP_HIDE_FROM_ABI void __flush(size_t __size_hint) {
    if (!__data_.__version_llvm_20_) {
      // LLVM-19 code path
      __flush_(__ptr_, __size_, reinterpret_cast<void*>(__data_.__obj_ << 1));
      __size_ = 0;
    } else {
      // LLVM-20 code path
      __prepare_write_(*this, __size_hint + 1); // + 1 to always have space for the next time
    }
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t __capacity() const { return __capacity_; }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t __size() const { return __size_; }

private:
  _CharT* __ptr_;
  size_t __capacity_;
  size_t __size_{0};
  union {
    // LLVM-19 member
    void (*__flush_)(_CharT*, size_t, void*);
    // LLVM-20 member
    void (*__prepare_write_)(__output_buffer<_CharT>&, size_t);
  };
  static_assert(sizeof(__flush_) == sizeof(__prepare_write_), "The union is an ABI break.");
  static_assert(alignof(decltype(__flush_)) == alignof(decltype(__prepare_write_)), "The union is an ABI break.");
  // Note this code is quite ugly, but it can cleaned up once the LLVM-19 parts
  // of the code are removed.
  union {
    struct {
      uintptr_t __version_llvm_20__ : 1;
      uintptr_t __obj_ : CHAR_BIT * sizeof(void*) - 1;
    };
    struct {
      uintptr_t __version_llvm_20_ : 1;
      uintptr_t __max_output_size_ : CHAR_BIT * sizeof(__max_output_size*) - 1;
    };
  } __data_;
  static_assert(sizeof(__data_) == sizeof(void*), "The struct is an ABI break.");
  static_assert(alignof(decltype(__data_)) == alignof(void*), "The struct is an ABI break.");

  /// Flushes the buffer when the output operation would overflow the buffer.
  ///
  /// A simple approach for the overflow detection would be something along the
  /// lines:
  /// \code
  /// // The internal buffer is large enough.
  /// if (__n <= __capacity_) {
  ///   // Flush when we really would overflow.
  ///   if (__size_ + __n >= __capacity_)
  ///     __flush();
  ///   ...
  /// }
  /// \endcode
  ///
  /// This approach works for all cases but one:
  /// A __format_to_n_buffer_base where \ref __enable_direct_output is true.
  /// In that case the \ref __capacity_ of the buffer changes during the first
  /// \ref __flush. During that operation the output buffer switches from its
  /// __writer_ to its __storage_. The \ref __capacity_ of the former depends
  /// on the value of n, of the latter is a fixed size. For example:
  /// - a format_to_n call with a 10'000 char buffer,
  /// - the buffer is filled with 9'500 chars,
  /// - adding 1'000 elements would overflow the buffer so the buffer gets
  ///   changed and the \ref __capacity_ decreases from 10'000 to
  ///   __buffer_size (256 at the time of writing).
  ///
  /// This means that the \ref __flush for this class may need to copy a part of
  /// the internal buffer to the proper output. In this example there will be
  /// 500 characters that need this copy operation.
  ///
  /// Note it would be more efficient to write 500 chars directly and then swap
  /// the buffers. This would make the code more complex and \ref format_to_n is
  /// not the most common use case. Therefore the optimization isn't done.
  _LIBCPP_HIDE_FROM_ABI void __flush_on_overflow(size_t __n) {
    __n += __size_;
    if (__n >= __capacity_)
      __flush(__n - __capacity_ + 1);
  }
};

// ***** ***** ***** LLVM-19 classes ***** ***** *****

/// A storage using an internal buffer.
///
/// This storage is used when writing a single element to the output iterator
/// is expensive.
template <__fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __internal_storage {
public:
  _LIBCPP_HIDE_FROM_ABI _CharT* __begin() { return __buffer_; }

  static constexpr size_t __buffer_size = 256 / sizeof(_CharT);

private:
  _CharT __buffer_[__buffer_size];
};

/// A storage writing directly to the storage.
///
/// This requires the storage to be a contiguous buffer of \a _CharT.
/// Since the output is directly written to the underlying storage this class
/// is just an empty class.
template <__fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __direct_storage {};

template <class _OutIt, class _CharT>
concept __enable_direct_output =
    __fmt_char_type<_CharT> &&
    (same_as<_OutIt, _CharT*>
     // TODO(hardening): the following check might not apply to hardened iterators and might need to be wrapped in an
     // `#ifdef`.
     || same_as<_OutIt, __wrap_iter<_CharT*>>);

/// Write policy for directly writing to the underlying output.
template <class _OutIt, __fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __writer_direct {
public:
  _LIBCPP_HIDE_FROM_ABI explicit __writer_direct(_OutIt __out_it) : __out_it_(__out_it) {}

  _LIBCPP_HIDE_FROM_ABI _OutIt __out_it() { return __out_it_; }

  _LIBCPP_HIDE_FROM_ABI void __flush(_CharT*, size_t __n) {
    // _OutIt can be a __wrap_iter<CharT*>. Therefore the original iterator
    // is adjusted.
    __out_it_ += __n;
  }

private:
  _OutIt __out_it_;
};

/// Write policy for copying the buffer to the output.
template <class _OutIt, __fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __writer_iterator {
public:
  _LIBCPP_HIDE_FROM_ABI explicit __writer_iterator(_OutIt __out_it) : __out_it_{std::move(__out_it)} {}

  _LIBCPP_HIDE_FROM_ABI _OutIt __out_it() && { return std::move(__out_it_); }

  _LIBCPP_HIDE_FROM_ABI void __flush(_CharT* __ptr, size_t __n) {
    __out_it_ = std::ranges::copy_n(__ptr, __n, std::move(__out_it_)).out;
  }

private:
  _OutIt __out_it_;
};

/// Concept to see whether a \a _Container is insertable.
///
/// The concept is used to validate whether multiple calls to a
/// \ref back_insert_iterator can be replace by a call to \c _Container::insert.
///
/// \note a \a _Container needs to opt-in to the concept by specializing
/// \ref __enable_insertable.
template <class _Container>
concept __insertable =
    __enable_insertable<_Container> && __fmt_char_type<typename _Container::value_type> &&
    requires(_Container& __t,
             add_pointer_t<typename _Container::value_type> __first,
             add_pointer_t<typename _Container::value_type> __last) { __t.insert(__t.end(), __first, __last); };

/// Extract the container type of a \ref back_insert_iterator.
template <class _It>
struct _LIBCPP_TEMPLATE_VIS __back_insert_iterator_container {
  using type = void;
};

template <__insertable _Container>
struct _LIBCPP_TEMPLATE_VIS __back_insert_iterator_container<back_insert_iterator<_Container>> {
  using type = _Container;
};

/// Write policy for inserting the buffer in a container.
template <class _Container>
class _LIBCPP_TEMPLATE_VIS __writer_container {
public:
  using _CharT = typename _Container::value_type;

  _LIBCPP_HIDE_FROM_ABI explicit __writer_container(back_insert_iterator<_Container> __out_it)
      : __container_{__out_it.__get_container()} {}

  _LIBCPP_HIDE_FROM_ABI auto __out_it() { return std::back_inserter(*__container_); }

  _LIBCPP_HIDE_FROM_ABI void __flush(_CharT* __ptr, size_t __n) {
    __container_->insert(__container_->end(), __ptr, __ptr + __n);
  }

private:
  _Container* __container_;
};

/// Selects the type of the writer used for the output iterator.
template <class _OutIt, class _CharT>
class _LIBCPP_TEMPLATE_VIS __writer_selector {
  using _Container = typename __back_insert_iterator_container<_OutIt>::type;

public:
  using type =
      conditional_t<!same_as<_Container, void>,
                    __writer_container<_Container>,
                    conditional_t<__enable_direct_output<_OutIt, _CharT>,
                                  __writer_direct<_OutIt, _CharT>,
                                  __writer_iterator<_OutIt, _CharT>>>;
};

/// The generic formatting buffer.
template <class _OutIt, __fmt_char_type _CharT>
  requires(output_iterator<_OutIt, const _CharT&>)
class _LIBCPP_TEMPLATE_VIS __format_buffer {
  using _Storage =
      conditional_t<__enable_direct_output<_OutIt, _CharT>, __direct_storage<_CharT>, __internal_storage<_CharT>>;

public:
  _LIBCPP_HIDE_FROM_ABI explicit __format_buffer(_OutIt __out_it)
    requires(same_as<_Storage, __internal_storage<_CharT>>)
      : __output_(__storage_.__begin(), __storage_.__buffer_size, this), __writer_(std::move(__out_it)) {}

  _LIBCPP_HIDE_FROM_ABI explicit __format_buffer(_OutIt __out_it)
    requires(same_as<_Storage, __direct_storage<_CharT>>)
      : __output_(std::__unwrap_iter(__out_it), size_t(-1), this), __writer_(std::move(__out_it)) {}

  _LIBCPP_HIDE_FROM_ABI auto __make_output_iterator() { return __output_.__make_output_iterator(); }

  _LIBCPP_HIDE_FROM_ABI void __flush(_CharT* __ptr, size_t __n) { __writer_.__flush(__ptr, __n); }

  _LIBCPP_HIDE_FROM_ABI _OutIt __out_it() && {
    __output_.__flush(0);
    return std::move(__writer_).__out_it();
  }

private:
  _LIBCPP_NO_UNIQUE_ADDRESS _Storage __storage_;
  __output_buffer<_CharT> __output_;
  typename __writer_selector<_OutIt, _CharT>::type __writer_;
};

// ***** ***** ***** LLVM-20 classes ***** ***** *****

// A dynamically growing buffer.
template <__fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __allocating_buffer : public __output_buffer<_CharT> {
public:
  __allocating_buffer(const __allocating_buffer&)            = delete;
  __allocating_buffer& operator=(const __allocating_buffer&) = delete;

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI __allocating_buffer() : __allocating_buffer{nullptr} {}

  [[nodiscard]]
  _LIBCPP_HIDE_FROM_ABI explicit __allocating_buffer(__max_output_size* __max_output_size)
      : __output_buffer<_CharT>{__buffer_, __buffer_size_, __prepare_write, __max_output_size} {}

  _LIBCPP_HIDE_FROM_ABI ~__allocating_buffer() {
    if (__ptr_ != __buffer_) {
      ranges::destroy_n(__ptr_, this->__size());
      allocator_traits<_Alloc>::deallocate(__alloc_, __ptr_, this->__capacity());
    }
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI basic_string_view<_CharT> __view() { return {__ptr_, this->__size()}; }

private:
  // At the moment the allocator is hard-code. There might be reasons to have
  // an allocator trait in the future. This ensures forward compatibility.
  using _Alloc = allocator<_CharT>;
  _LIBCPP_NO_UNIQUE_ADDRESS _Alloc __alloc_;

  // Since allocating is expensive the class has a small internal buffer. When
  // its capacity is exceeded a dynamic buffer will be allocated.
  static constexpr size_t __buffer_size_ = 256;
  _CharT __buffer_[__buffer_size_];

  _CharT* __ptr_{__buffer_};

  _LIBCPP_HIDE_FROM_ABI void __grow_buffer(size_t __capacity) {
    if (__capacity < __buffer_size_)
      return;

    _LIBCPP_ASSERT_INTERNAL(__capacity > this->__capacity(), "the buffer must grow");
    auto __result = std::__allocate_at_least(__alloc_, __capacity);
    auto __guard  = std::__make_exception_guard([&] {
      allocator_traits<_Alloc>::deallocate(__alloc_, __result.ptr, __result.count);
    });
    // This shouldn't throw, but just to be safe. Note that at -O1 this
    // guard is optimized away so there is no runtime overhead.
    new (__result.ptr) _CharT[__result.count];
    std::copy_n(__ptr_, this->__size(), __result.ptr);
    __guard.__complete();
    if (__ptr_ != __buffer_) {
      ranges::destroy_n(__ptr_, this->__capacity());
      allocator_traits<_Alloc>::deallocate(__alloc_, __ptr_, this->__capacity());
    }

    __ptr_ = __result.ptr;
    this->__buffer_moved(__ptr_, __result.count);
  }

  _LIBCPP_HIDE_FROM_ABI void __prepare_write(size_t __size_hint) {
    __grow_buffer(std::max<size_t>(this->__capacity() + __size_hint, this->__capacity() * 1.6));
  }

  _LIBCPP_HIDE_FROM_ABI static void __prepare_write(__output_buffer<_CharT>& __buffer, size_t __size_hint) {
    static_cast<__allocating_buffer<_CharT>&>(__buffer).__prepare_write(__size_hint);
  }
};

// A buffer that directly writes to the underlying buffer.
template <class _OutIt, __fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __direct_iterator_buffer : public __output_buffer<_CharT> {
public:
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI explicit __direct_iterator_buffer(_OutIt __out_it)
      : __direct_iterator_buffer{__out_it, nullptr} {}

  [[nodiscard]]
  _LIBCPP_HIDE_FROM_ABI explicit __direct_iterator_buffer(_OutIt __out_it, __max_output_size* __max_output_size)
      : __output_buffer<_CharT>{std::__unwrap_iter(__out_it), __buffer_size, __prepare_write, __max_output_size},
        __out_it_(__out_it) {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI _OutIt __out_it() && { return __out_it_ + this->__size(); }

private:
  // The function format_to expects a buffer large enough for the output. The
  // function format_to_n has its own helper class that restricts the number of
  // write options. So this function class can pretend to have an infinite
  // buffer.
  static constexpr size_t __buffer_size = -1;

  _OutIt __out_it_;

  _LIBCPP_HIDE_FROM_ABI static void
  __prepare_write([[maybe_unused]] __output_buffer<_CharT>& __buffer, [[maybe_unused]] size_t __size_hint) {
    std::__throw_length_error("__direct_iterator_buffer");
  }
};

// A buffer that writes its output to the end of a container.
template <class _OutIt, __fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __container_inserter_buffer : public __output_buffer<_CharT> {
public:
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI explicit __container_inserter_buffer(_OutIt __out_it)
      : __container_inserter_buffer{__out_it, nullptr} {}

  [[nodiscard]]
  _LIBCPP_HIDE_FROM_ABI explicit __container_inserter_buffer(_OutIt __out_it, __max_output_size* __max_output_size)
      : __output_buffer<_CharT>{__buffer_, __buffer_size, __prepare_write, __max_output_size},
        __container_{__out_it.__get_container()} {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto __out_it() && {
    __container_->insert(__container_->end(), __buffer_, __buffer_ + this->__size());
    return std::back_inserter(*__container_);
  }

private:
  typename __back_insert_iterator_container<_OutIt>::type* __container_;

  // This class uses a fixed size buffer and appends the elements in
  // __buffer_size chunks. An alternative would be to use an allocating buffer
  // and append the output in one write operation. Benchmarking showed no
  // performance difference.
  static constexpr size_t __buffer_size = 256;
  _CharT __buffer_[__buffer_size];

  _LIBCPP_HIDE_FROM_ABI void __prepare_write() {
    __container_->insert(__container_->end(), __buffer_, __buffer_ + this->__size());
    this->__buffer_flused();
  }

  _LIBCPP_HIDE_FROM_ABI static void
  __prepare_write(__output_buffer<_CharT>& __buffer, [[maybe_unused]] size_t __size_hint) {
    static_cast<__container_inserter_buffer<_OutIt, _CharT>&>(__buffer).__prepare_write();
  }
};

// A buffer that writes to an iterator.
//
// Unlinke the __container_inserter_buffer this class' perfomance does benefit
// from allocating and then inserting.
template <class _OutIt, __fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __iterator_buffer : public __allocating_buffer<_CharT> {
public:
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI explicit __iterator_buffer(_OutIt __out_it)
      : __allocating_buffer<_CharT>{}, __out_it_{std::move(__out_it)} {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI explicit __iterator_buffer(_OutIt __out_it, __max_output_size* __max_output_size)
      : __allocating_buffer<_CharT>{__max_output_size}, __out_it_{std::move(__out_it)} {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto __out_it() && {
    return std::ranges::copy(this->__view(), std::move(__out_it_)).out;
  }

private:
  _OutIt __out_it_;
};

// Selects the type of the buffer used for the output iterator.
template <class _OutIt, __fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __buffer_selector {
  using _Container = __back_insert_iterator_container<_OutIt>::type;

public:
  using type =
      conditional_t<!same_as<_Container, void>,
                    __container_inserter_buffer<_OutIt, _CharT>,
                    conditional_t<__enable_direct_output<_OutIt, _CharT>,
                                  __direct_iterator_buffer<_OutIt, _CharT>,
                                  __iterator_buffer<_OutIt, _CharT>>>;
};

// A buffer that counts and limits the number of insertions.
template <class _OutIt, __fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __format_to_n_buffer : private __buffer_selector<_OutIt, _CharT>::type {
public:
  using _Base = __buffer_selector<_OutIt, _CharT>::type;

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI __format_to_n_buffer(_OutIt __out_it, iter_difference_t<_OutIt> __n)
      : _Base{std::move(__out_it), std::addressof(__max_output_size_)},
        __max_output_size_{__n < 0 ? size_t{0} : static_cast<size_t>(__n)} {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto __make_output_iterator() { return _Base::__make_output_iterator(); }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI format_to_n_result<_OutIt> __result() && {
    return {static_cast<_Base&&>(*this).__out_it(),
            static_cast<iter_difference_t<_OutIt>>(__max_output_size_.__code_units_written())};
  }

private:
  __max_output_size __max_output_size_;
};

// A buffer that counts the number of insertions.
//
// Since formatted_size only needs to know the size, the output itself is
// discarded.
template <__fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __formatted_size_buffer : private __output_buffer<_CharT> {
public:
  using _Base = __output_buffer<_CharT>;

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI __formatted_size_buffer()
      : _Base{nullptr, 0, __prepare_write, std::addressof(__max_output_size_)} {}

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI auto __make_output_iterator() { return _Base::__make_output_iterator(); }

  // This function does not need to be r-value qualified, however this is
  // consistent with similar objects.
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI size_t __result() && { return __max_output_size_.__code_units_written(); }

private:
  __max_output_size __max_output_size_{0};

  _LIBCPP_HIDE_FROM_ABI static void
  __prepare_write([[maybe_unused]] __output_buffer<_CharT>& __buffer, [[maybe_unused]] size_t __size_hint) {
    // Note this function does not satisfy the requirement of giving a 1 code unit buffer.
    _LIBCPP_ASSERT_INTERNAL(
        false, "Since __max_output_size_.__max_size_ == 0 there should never be call to this function.");
  }
};

// ***** ***** ***** LLVM-19 and LLVM-20 class ***** ***** *****

// A dynamically growing buffer intended to be used for retargeting a context.
//
// P2286 Formatting ranges adds range formatting support. It allows the user to
// specify the minimum width for the entire formatted range.  The width of the
// range is not known until the range is formatted. Formatting is done to an
// output_iterator so there's no guarantee it would be possible to add the fill
// to the front of the output. Instead the range is formatted to a temporary
// buffer and that buffer is formatted as a string.
//
// There is an issue with that approach, the format context used in
// std::formatter<T>::format contains the output iterator used as part of its
// type. So using this output iterator means there needs to be a new format
// context and the format arguments need to be retargeted to the new context.
// This retargeting is done by a basic_format_context specialized for the
// __iterator of this container.
//
// This class uses its own buffer management, since using vector
// would lead to a circular include with formatter for vector<bool>.
template <__fmt_char_type _CharT>
class _LIBCPP_TEMPLATE_VIS __retarget_buffer {
  using _Alloc = allocator<_CharT>;

public:
  using value_type = _CharT;

  struct __iterator {
    using difference_type = ptrdiff_t;
    using value_type      = _CharT;

    _LIBCPP_HIDE_FROM_ABI constexpr explicit __iterator(__retarget_buffer& __buffer)
        : __buffer_(std::addressof(__buffer)) {}
    _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator=(const _CharT& __c) {
      __buffer_->push_back(__c);
      return *this;
    }
    _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator=(_CharT&& __c) {
      __buffer_->push_back(__c);
      return *this;
    }

    _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator*() { return *this; }
    _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() { return *this; }
    _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int) { return *this; }
    __retarget_buffer* __buffer_;
  };

  __retarget_buffer(const __retarget_buffer&)            = delete;
  __retarget_buffer& operator=(const __retarget_buffer&) = delete;

  _LIBCPP_HIDE_FROM_ABI explicit __retarget_buffer(size_t __size_hint) {
    // When the initial size is very small a lot of resizes happen
    // when elements added. So use a hard-coded minimum size.
    //
    // Note a size < 2 will not work
    // - 0 there is no buffer, while push_back requires 1 empty element.
    // - 1 multiplied by the grow factor is 1 and thus the buffer never
    //   grows.
    auto __result = std::__allocate_at_least(__alloc_, std::max(__size_hint, 256 / sizeof(_CharT)));
    __ptr_        = __result.ptr;
    __capacity_   = __result.count;
  }

  _LIBCPP_HIDE_FROM_ABI ~__retarget_buffer() {
    ranges::destroy_n(__ptr_, __size_);
    allocator_traits<_Alloc>::deallocate(__alloc_, __ptr_, __capacity_);
  }

  _LIBCPP_HIDE_FROM_ABI __iterator __make_output_iterator() { return __iterator{*this}; }

  _LIBCPP_HIDE_FROM_ABI void push_back(_CharT __c) {
    std::construct_at(__ptr_ + __size_, __c);
    ++__size_;

    if (__size_ == __capacity_)
      __grow_buffer();
  }

  template <__fmt_char_type _InCharT>
  _LIBCPP_HIDE_FROM_ABI void __copy(basic_string_view<_InCharT> __str) {
    size_t __n = __str.size();
    if (__size_ + __n >= __capacity_)
      // Push_back requires the buffer to have room for at least one character.
      __grow_buffer(__size_ + __n + 1);

    std::uninitialized_copy_n(__str.data(), __n, __ptr_ + __size_);
    __size_ += __n;
  }

  template <contiguous_iterator _Iterator,
            class _UnaryOperation,
            __fmt_char_type _InCharT = typename iterator_traits<_Iterator>::value_type>
  _LIBCPP_HIDE_FROM_ABI void __transform(_Iterator __first, _Iterator __last, _UnaryOperation __operation) {
    _LIBCPP_ASSERT_INTERNAL(__first <= __last, "not a valid range");

    size_t __n = static_cast<size_t>(__last - __first);
    if (__size_ + __n >= __capacity_)
      // Push_back requires the buffer to have room for at least one character.
      __grow_buffer(__size_ + __n + 1);

    std::uninitialized_default_construct_n(__ptr_ + __size_, __n);
    std::transform(__first, __last, __ptr_ + __size_, std::move(__operation));
    __size_ += __n;
  }

  _LIBCPP_HIDE_FROM_ABI void __fill(size_t __n, _CharT __value) {
    if (__size_ + __n >= __capacity_)
      // Push_back requires the buffer to have room for at least one character.
      __grow_buffer(__size_ + __n + 1);

    std::uninitialized_fill_n(__ptr_ + __size_, __n, __value);
    __size_ += __n;
  }

  _LIBCPP_HIDE_FROM_ABI basic_string_view<_CharT> __view() { return {__ptr_, __size_}; }

private:
  _LIBCPP_HIDE_FROM_ABI void __grow_buffer() { __grow_buffer(__capacity_ * 1.6); }

  _LIBCPP_HIDE_FROM_ABI void __grow_buffer(size_t __capacity) {
    _LIBCPP_ASSERT_INTERNAL(__capacity > __capacity_, "the buffer must grow");
    auto __result = std::__allocate_at_least(__alloc_, __capacity);
    auto __guard  = std::__make_exception_guard([&] {
      allocator_traits<_Alloc>::deallocate(__alloc_, __result.ptr, __result.count);
    });
    // This shouldn't throw, but just to be safe. Note that at -O1 this
    // guard is optimized away so there is no runtime overhead.
    std::uninitialized_move_n(__ptr_, __size_, __result.ptr);
    __guard.__complete();
    ranges::destroy_n(__ptr_, __size_);
    allocator_traits<_Alloc>::deallocate(__alloc_, __ptr_, __capacity_);

    __ptr_      = __result.ptr;
    __capacity_ = __result.count;
  }
  _LIBCPP_NO_UNIQUE_ADDRESS _Alloc __alloc_;
  _CharT* __ptr_;
  size_t __capacity_;
  size_t __size_{0};
};

} // namespace __format

#endif //_LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___FORMAT_BUFFER_H
