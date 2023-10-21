// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HAVE_DEPENDENT_EH_ABI
#error this header may only be used with libc++abi or libcxxrt
#endif

#if defined(LIBCXXRT)
extern "C" {
    // Although libcxxrt defines these two (as an ABI-library should),
    // it doesn't declare them in some versions.
    void *__cxa_allocate_exception(size_t thrown_size);
    void __cxa_free_exception(void* thrown_exception);

    __attribute__((weak)) __cxa_exception *__cxa_init_primary_exception(void *, std::type_info *, void(*)(void*));
}
#else
extern "C" {
    __attribute__((weak)) __cxa_exception *__cxa_init_primary_exception(void *, std::type_info *, void(_LIBCXXABI_DTOR_FUNC*)(void*)) throw();
}
#endif

namespace std {

exception_ptr::~exception_ptr() noexcept
{
    __cxa_decrement_exception_refcount(__ptr_);
}

exception_ptr::exception_ptr(const exception_ptr& other) noexcept
    : __ptr_(other.__ptr_)
{
    __cxa_increment_exception_refcount(__ptr_);
}

exception_ptr& exception_ptr::operator=(const exception_ptr& other) noexcept
{
    if (__ptr_ != other.__ptr_)
    {
        __cxa_increment_exception_refcount(other.__ptr_);
        __cxa_decrement_exception_refcount(__ptr_);
        __ptr_ = other.__ptr_;
    }
    return *this;
}

#  ifndef _LIBCPP_HAS_NO_EXCEPTIONS
#    if !defined(_LIBCPP_HAS_NO_RTTI)
void *exception_ptr::__init_native_exception(size_t size, type_info *tinfo, void (*dest)(void *)) noexcept
{
    __cxa_exception *(*cxa_init_primary_exception_fn)(void *, std::type_info *, void(_LIBCXXABI_DTOR_FUNC*)(void*)) = __cxa_init_primary_exception;
    if (cxa_init_primary_exception_fn != nullptr) {
        void *__ex = __cxa_allocate_exception(size);
        (void)cxa_init_primary_exception_fn(__ex, tinfo, dest);
        return __ex;
    }
    else {
        return nullptr;
    }
}

void exception_ptr::__free_native_exception(void *thrown_object) noexcept
{
    __cxa_free_exception(thrown_object);
}

exception_ptr exception_ptr::__from_native_exception_pointer(void *__e) noexcept {
    exception_ptr ptr;
    ptr.__ptr_ = __e;
    __cxa_increment_exception_refcount(ptr.__ptr_);

    return ptr;
}
#    endif
#  endif

nested_exception::nested_exception() noexcept
    : __ptr_(current_exception())
{
}

nested_exception::~nested_exception() noexcept
{
}

_LIBCPP_NORETURN
void
nested_exception::rethrow_nested() const
{
    if (__ptr_ == nullptr)
        terminate();
    rethrow_exception(__ptr_);
}

exception_ptr current_exception() noexcept
{
    // be nicer if there was a constructor that took a ptr, then
    // this whole function would be just:
    //    return exception_ptr(__cxa_current_primary_exception());
    exception_ptr ptr;
    ptr.__ptr_ = __cxa_current_primary_exception();
    return ptr;
}

_LIBCPP_NORETURN
void rethrow_exception(exception_ptr p)
{
    __cxa_rethrow_primary_exception(p.__ptr_);
    // if p.__ptr_ is NULL, above returns so we terminate
    terminate();
}

} // namespace std
