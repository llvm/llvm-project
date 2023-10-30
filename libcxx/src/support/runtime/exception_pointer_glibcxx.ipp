// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// libsupc++ does not implement the dependent EH ABI and the functionality
// it uses to implement std::exception_ptr (which it declares as an alias of
// std::__exception_ptr::exception_ptr) is not directly exported to clients. So
// we have little choice but to hijack std::__exception_ptr::exception_ptr's
// (which fortunately has the same layout as our std::exception_ptr) copy
// constructor, assignment operator and destructor (which are part of its
// stable ABI), and its rethrow_exception(std::__exception_ptr::exception_ptr)
// function.


#  if defined(_LIBCPP_EXCEPTION_PTR_DIRECT_INIT)
extern "C"
{
    void* __cxa_allocate_exception(size_t) throw();
    void __cxa_free_exception(void*) throw();

    _LIBCPP_WEAK void* __cxa_init_primary_exception(void*, std::type_info*, void(_LIBCXX_DTOR_FUNC*)(void*)) throw();
}
#  endif

namespace std {

namespace __exception_ptr
{

struct exception_ptr
{
    void* __ptr_;

#  if defined(_LIBCPP_EXCEPTION_PTR_DIRECT_INIT)
    explicit exception_ptr(void*) noexcept;
#  endif
    exception_ptr(const exception_ptr&) noexcept;
    exception_ptr& operator=(const exception_ptr&) noexcept;
    ~exception_ptr() noexcept;
};

}

_LIBCPP_NORETURN void rethrow_exception(__exception_ptr::exception_ptr);

exception_ptr::~exception_ptr() noexcept
{
    reinterpret_cast<__exception_ptr::exception_ptr*>(this)->~exception_ptr();
}

exception_ptr::exception_ptr(const exception_ptr& other) noexcept
    : __ptr_(other.__ptr_)
{
    new (reinterpret_cast<void*>(this)) __exception_ptr::exception_ptr(
        reinterpret_cast<const __exception_ptr::exception_ptr&>(other));
}

exception_ptr& exception_ptr::operator=(const exception_ptr& other) noexcept
{
    *reinterpret_cast<__exception_ptr::exception_ptr*>(this) =
        reinterpret_cast<const __exception_ptr::exception_ptr&>(other);
    return *this;
}

#  if defined(_LIBCPP_EXCEPTION_PTR_DIRECT_INIT)
void *exception_ptr::__init_native_exception(size_t size, type_info* tinfo, void (_LIBCXX_DTOR_FUNC* dest)(void*)) noexcept
{
    void *(*cxa_init_primary_exception_fn)(void*, std::type_info*, void(_LIBCXX_DTOR_FUNC*)(void*)) = __cxa_init_primary_exception;
    if (cxa_init_primary_exception_fn != nullptr)
    {
        void* __ex = __cxa_allocate_exception(size);
        (void)cxa_init_primary_exception_fn(__ex, tinfo, dest);
        return __ex;
    }
    else
    {
        return nullptr;
    }
}

void exception_ptr::__free_native_exception(void* thrown_object) noexcept
{
    __cxa_free_exception(thrown_object);
}

exception_ptr exception_ptr::__from_native_exception_pointer(void* __e) noexcept
{
    exception_ptr ptr{};
    new (reinterpret_cast<void*>(&ptr)) __exception_ptr::exception_ptr(__e);

    return ptr;
}
#  endif

nested_exception::nested_exception() noexcept
    : __ptr_(current_exception())
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

_LIBCPP_NORETURN
void rethrow_exception(exception_ptr p)
{
    rethrow_exception(reinterpret_cast<__exception_ptr::exception_ptr&>(p));
}

} // namespace std
