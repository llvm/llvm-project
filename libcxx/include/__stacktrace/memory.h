// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_CSTR_BUFFER_H
#define _LIBCPP_STACKTRACE_CSTR_BUFFER_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

#  include <__assert>
#  include <cstddef>
#  include <cstdint>
#  include <functional>
#  include <string>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

/*
A few memory-related utilities:

  * An `arena` which provides allocator-like / malloc-like functionality for us,
    for objects used internally as well as objects returned to the caller.
    Uses the caller's provided allocator, so none of these involve heap allocations
    "outside of" the caller-provided allocator.
  * A `str` class, inheriting from `std::string`, ensuring allocations happen via `arena`
  * A `fixed_str` class, not related to the arena, but instead backed by a `char[n]`
    array within that same struct, so it doesn't perform any [de]allocations; it only
    uses its own character array.

A small amount of glue / hacks are done here to allow the rest of the stacktrace-related
code to use familiar string etc. operations, while encapsulating away the details of where
memory might come from, since we need to be careful about unexpected allocations.
*/

// clang-format off

struct byte_pool final {
  byte* ptr_;
  function<void()> destroy_;
  byte_pool* link_;
  byte* end_;

  byte_pool(byte* __bytes,
            size_t __size,
            function<void()> __destroy = [] {},
            byte_pool* __link = nullptr) noexcept
    : ptr_(__bytes), destroy_(__destroy), link_(__link), end_(__bytes + __size) {}

  byte* operator()(size_t __sz, size_t __align) noexcept {
    auto __ptr = uintptr_t(ptr_);                         // convert curr ptr to integer, to do math
    auto __misalign = __ptr % __align;                    // if current ptr not aligned,
    if (__misalign) { __ptr += (__align - __misalign); }  // waste a few bytes to ensure alignment
    auto __ret = __ptr;                                   // we would return this aligned position
    __ptr += __sz;                                        // next object will start here
    if (__ptr > uintptr_t(end_)) { return nullptr; }      // if this exceeds our space, then fail
    ptr_ = (byte*) __ptr;                                 // otherwise update current position
    return (byte*) __ret;                                 // returned aligned position as byte ptr
  }
};

template <size_t _Sz>
struct stack_bytes final {
  byte bytes_[_Sz];

  ~stack_bytes() = default;
  stack_bytes() noexcept = default;
  stack_bytes(const stack_bytes&) = delete;
  stack_bytes(stack_bytes&&) = delete;

  byte_pool pool() { return {bytes_, _Sz, []{}, nullptr}; }
};

struct arena {
  function<byte*(size_t)> new_bytes_;         // new byte-array factory
  function<void(void*, size_t)> del_bytes_;   // byte-array destroyer
  byte_pool* curr_pool_;                      // byte pool currently "in effect"
  byte_pool* next_pool_;                      // allocated (from curr_pool_) but not initialized
  size_t allocs_ {};                          // number of successful allocations
  size_t deallocs_ {};                        // incremented on each dealloc; dtor ensures these are equal!

  // An arena is scoped to a `basic_stacktrace::current` invocation, so this is usable by only one thread.
  // Additionally, it's used internally throughout many function calls, so for convenience, store it here.
  // Also avoids the need for state inside `alloc`, since it can use this pointer instead of an internal one.
  static thread_local arena* active_arena_ptr_;

  static arena& get_active() {
    auto* __ret = active_arena_ptr_;
    _LIBCPP_ASSERT(__ret, "no active arena for this thread");
    return *__ret;
  }

  ~arena() {
    _LIBCPP_ASSERT(active_arena_ptr_ == this, "different arena unexpectively set as the active one");
    active_arena_ptr_ = nullptr;
    _LIBCPP_ASSERT(deallocs_ == allocs_, "destructed arena still has live objects");
    while (curr_pool_) { curr_pool_->destroy_(); curr_pool_ = curr_pool_->link_; }
  }

  arena(auto&& __new_bytes, auto&& __del_bytes, byte_pool& __initial_pool) noexcept
    : new_bytes_(__new_bytes), del_bytes_(__del_bytes), curr_pool_(&__initial_pool) {
    prep_next_pool();
    _LIBCPP_ASSERT(!active_arena_ptr_, "already an active arena");
    active_arena_ptr_ = this;
  }

  template <class _UA>
  static auto as_byte_alloc(_UA const& __user_alloc) {
    return (typename allocator_traits<_UA>::template rebind_alloc<std::byte>)(__user_alloc);
  }

  template <class _UA>
  arena(byte_pool& __initial_pool, _UA const& __user_alloc)
    : arena(
      [&__user_alloc] (size_t __sz) { return as_byte_alloc(__user_alloc).allocate(__sz); },
      [&__user_alloc] (void* __ptr, size_t __sz) { return as_byte_alloc(__user_alloc).deallocate((byte*)__ptr, __sz); },
      __initial_pool) {}

  arena(arena const&)            = delete;
  arena& operator=(arena const&) = delete;

  void prep_next_pool() noexcept {
    // Allocate (via current pool) a new byte_pool record, while we have enough space.
    // When the current pool runs out of space, this one will be ready to use.
    next_pool_ = (byte_pool*) (*curr_pool_)(sizeof(byte_pool), alignof(byte_pool));
    _LIBCPP_ASSERT(next_pool_, "could not allocate next pool");
  }

  void expand(size_t __atleast) noexcept {
    constexpr static size_t __k_default_new_pool = 1 << 12;
    auto __size = max(__atleast, __k_default_new_pool);
    // "next_pool_" was already allocated, just need to initialize it
    auto* __bytes = new_bytes_(__size);
    _LIBCPP_ASSERT(__bytes, "could not allocate more bytes for arena");
    curr_pool_ = new (next_pool_) byte_pool(__bytes, __size, [=, this] { del_bytes_(__bytes, __size); }, curr_pool_);
    prep_next_pool();
  }

  /** Does nothing; all memory is released when arena is destroyed. */
  void dealloc(std::byte*, size_t) noexcept { ++deallocs_; }

  std::byte* alloc(size_t __size, size_t __align) noexcept {
    auto* __ret = (*curr_pool_)(__size, __align);
    if (__ret) [[likely]] { goto success; }
    // Need a new pool to accommodate this request + internal structs
    expand(__size + __align + sizeof(byte_pool) + alignof(byte_pool)); // upper bound
    __ret = (*curr_pool_)(__size, __align);
    _LIBCPP_ASSERT(__ret, "arena failed to allocate");
success:
    ++allocs_;
    return __ret;
  }
};

template <typename _Tp>
struct alloc {
  using value_type = _Tp;

  _Tp* allocate(size_t __n) {
    auto& __arena = arena::get_active();
    return (_Tp*)__arena.alloc(__n * sizeof(_Tp), alignof(_Tp));
  }

  void deallocate(_Tp* __ptr, size_t __n) {
    auto& __arena = arena::get_active();
    __arena.dealloc((std::byte*)__ptr, __n * sizeof(_Tp));
  }
};

template <typename _Tp, size_t _Sz>
struct fixed_buf {
  using value_type = _Tp;
  template <typename _Up> struct rebind { using other = fixed_buf<_Up, _Sz>; };

  _Tp __buf_[_Sz];
  size_t __size_;
  void deallocate(_Tp*, size_t) {}
  _Tp* allocate(size_t) { return __buf_; }
};

template <size_t _Sz>
struct fixed_str : std::basic_string<char, std::char_traits<char>, fixed_buf<char, _Sz>> {
  using _Base _LIBCPP_NODEBUG = std::basic_string<char, std::char_traits<char>, fixed_buf<char, _Sz>>;
  using _Base::operator=;

  fixed_buf<char, _Sz> __fb_;
  fixed_str() : _Base(__fb_) {
    this->resize(_Sz - 1);
    this->resize(0);
  }
  fixed_str(fixed_str const& __rhs) : fixed_str() { _Base::operator=(__rhs); }
  fixed_str& operator=(fixed_str const& __rhs) = default;
};

struct str : std::basic_string<char, std::char_traits<char>, alloc<char>> {
  using _Base _LIBCPP_NODEBUG = std::basic_string<char, std::char_traits<char>, alloc<char>>;
  using _Base::basic_string;
  using _Base::operator=;

  bool valid() const { return data() != nullptr; }

  operator bool() const { return valid() && !empty(); }

  template <typename... _AL>
  static str makef(char const* __fmt, _AL&&... __args) {
    str __ret{};

#  ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wformat-security"
#    pragma clang diagnostic ignored "-Wformat-nonliteral"
#  endif
    auto __need = std::snprintf(nullptr, 0, __fmt, __args...);
    __ret.resize_and_overwrite(__need + 1, [&](char* __data, size_t __size) {
      return std::snprintf(__data, __size + 1, __fmt, std::forward<_AL>(__args)...);
    });
#  ifdef __clang__
#    pragma clang diagnostic pop
#  endif

    return __ret;
  }
};

// clang-format on

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP_STACKTRACE_CSTR_BUFFER_H
