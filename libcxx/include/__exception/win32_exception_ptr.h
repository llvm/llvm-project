//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// exception standard header

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _EXCEPTION_
#define _EXCEPTION_
#include <yvals.h>
#if _STL_COMPILER_PREPROCESSOR

#include <cstdlib>
#include <type_traits>

#pragma pack(push, _CRT_PACKING)
#pragma warning(push, _STL_WARNING_LEVEL)
#pragma warning(disable : _STL_DISABLED_WARNINGS)
_STL_DISABLE_CLANG_WARNINGS
#pragma push_macro("new")
#undef new

_STD_BEGIN

#if _HAS_DEPRECATED_UNCAUGHT_EXCEPTION
_EXPORT_STD extern "C++" _CXX17_DEPRECATE_UNCAUGHT_EXCEPTION _NODISCARD _CRTIMP2_PURE bool __CLRCALL_PURE_OR_CDECL
    uncaught_exception() noexcept;
#endif // _HAS_DEPRECATED_UNCAUGHT_EXCEPTION
_EXPORT_STD extern "C++" _NODISCARD _CRTIMP2_PURE int __CLRCALL_PURE_OR_CDECL uncaught_exceptions() noexcept;

_STD_END

#if _HAS_EXCEPTIONS

#include <malloc.h> // TRANSITION, VSO-2048380: This is unnecessary, but many projects assume it as of 2024-04-29
#include <vcruntime_exception.h>

_STD_BEGIN

_EXPORT_STD using ::terminate;

#ifndef _M_CEE_PURE
_EXPORT_STD using ::set_terminate;
_EXPORT_STD using ::terminate_handler;

_EXPORT_STD _NODISCARD inline terminate_handler __CRTDECL get_terminate() noexcept {
    // get current terminate handler
    return _get_terminate();
}
#endif // !defined(_M_CEE_PURE)

#if _HAS_UNEXPECTED
using ::unexpected;

#ifndef _M_CEE_PURE
using ::set_unexpected;
using ::unexpected_handler;

_NODISCARD inline unexpected_handler __CRTDECL get_unexpected() noexcept {
    // get current unexpected handler
    return _get_unexpected();
}
#endif // !defined(_M_CEE_PURE)
#endif // _HAS_UNEXPECTED

_STD_END

#else // ^^^ _HAS_EXCEPTIONS / !_HAS_EXCEPTIONS vvv

#pragma push_macro("stdext")
#undef stdext

_STDEXT_BEGIN
class exception;
_STDEXT_END

_STD_BEGIN

_EXPORT_STD using _STDEXT exception;

using _Prhand = void(__cdecl*)(const exception&);

extern _CRTIMP2_PURE_IMPORT _Prhand _Raise_handler; // pointer to raise handler

_STD_END

_STDEXT_BEGIN
class exception { // base of all library exceptions
public:
    static _STD _Prhand _Set_raise_handler(_STD _Prhand _Pnew) { // register a handler for _Raise calls
        const _STD _Prhand _Pold = _STD _Raise_handler;
        _STD _Raise_handler      = _Pnew;
        return _Pold;
    }

    // this constructor is necessary to compile
    // successfully header new for _HAS_EXCEPTIONS==0 scenario
    explicit __CLR_OR_THIS_CALL exception(const char* _Message = "unknown", int = 1) noexcept : _Ptr(_Message) {}

    __CLR_OR_THIS_CALL exception(const exception& _Right) noexcept : _Ptr(_Right._Ptr) {}

    exception& __CLR_OR_THIS_CALL operator=(const exception& _Right) noexcept {
        _Ptr = _Right._Ptr;
        return *this;
    }

    virtual __CLR_OR_THIS_CALL ~exception() noexcept {}

    _NODISCARD virtual const char* __CLR_OR_THIS_CALL what() const noexcept { // return pointer to message string
        return _Ptr ? _Ptr : "unknown exception";
    }

    [[noreturn]] void __CLR_OR_THIS_CALL _Raise() const { // raise the exception
        if (_STD _Raise_handler) {
            (*_STD _Raise_handler)(*this); // call raise handler if present
        }

        _Doraise(); // call the protected virtual
        _RAISE(*this); // raise this exception
    }

protected:
    virtual void __CLR_OR_THIS_CALL _Doraise() const {} // perform class-specific exception handling

    const char* _Ptr; // the message pointer
};

class bad_exception : public exception { // base of all bad exceptions
public:
    __CLR_OR_THIS_CALL bad_exception(const char* _Message = "bad exception") noexcept : exception(_Message) {}

    __CLR_OR_THIS_CALL ~bad_exception() noexcept override {}

protected:
    void __CLR_OR_THIS_CALL _Doraise() const override { // raise this exception
        _RAISE(*this);
    }
};

class bad_array_new_length;

class bad_alloc : public exception { // base of all bad allocation exceptions
public:
    __CLR_OR_THIS_CALL bad_alloc() noexcept
        : exception("bad allocation", 1) {} // construct from message string with no memory allocation

    __CLR_OR_THIS_CALL ~bad_alloc() noexcept override {}

private:
    friend bad_array_new_length;

    __CLR_OR_THIS_CALL bad_alloc(const char* _Message) noexcept
        : exception(_Message, 1) {} // construct from message string with no memory allocation

protected:
    void __CLR_OR_THIS_CALL _Doraise() const override { // perform class-specific exception handling
        _RAISE(*this);
    }
};

class bad_array_new_length : public bad_alloc {
public:
    bad_array_new_length() noexcept : bad_alloc("bad array new length") {}
};

_STDEXT_END

_STD_BEGIN
_EXPORT_STD using terminate_handler = void(__cdecl*)();

_EXPORT_STD inline terminate_handler __CRTDECL set_terminate(terminate_handler) noexcept {
    // register a terminate handler
    return nullptr;
}

_EXPORT_STD [[noreturn]] inline void __CRTDECL terminate() noexcept {
    // handle exception termination
    _CSTD abort();
}

_EXPORT_STD _NODISCARD inline terminate_handler __CRTDECL get_terminate() noexcept {
    // get current terminate handler
    return nullptr;
}

#if _HAS_UNEXPECTED
using unexpected_handler = void(__cdecl*)();

inline unexpected_handler __CRTDECL set_unexpected(unexpected_handler) noexcept {
    // register an unexpected handler
    return nullptr;
}

inline void __CRTDECL unexpected() {} // handle unexpected exception

_NODISCARD inline unexpected_handler __CRTDECL get_unexpected() noexcept {
    // get current unexpected handler
    return nullptr;
}
#endif // _HAS_UNEXPECTED

_EXPORT_STD using _STDEXT bad_alloc;
_EXPORT_STD using _STDEXT bad_array_new_length;
_EXPORT_STD using _STDEXT bad_exception;

_STD_END

#pragma pop_macro("stdext")

#endif // ^^^ !_HAS_EXCEPTIONS ^^^

extern "C++" _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCreate(_Out_ void*) noexcept;
extern "C++" _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrDestroy(_Inout_ void*) noexcept;
extern "C++" _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCopy(_Out_ void*, _In_ const void*) noexcept;
extern "C++" _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrAssign(_Inout_ void*, _In_ const void*) noexcept;
extern "C++" _CRTIMP2_PURE bool __CLRCALL_PURE_OR_CDECL __ExceptionPtrCompare(
    _In_ const void*, _In_ const void*) noexcept;
extern "C++" _CRTIMP2_PURE bool __CLRCALL_PURE_OR_CDECL __ExceptionPtrToBool(_In_ const void*) noexcept;
extern "C++" _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrSwap(_Inout_ void*, _Inout_ void*) noexcept;
extern "C++" _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCurrentException(void*) noexcept;
extern "C++" [[noreturn]] _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrRethrow(_In_ const void*);
extern "C++" _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCopyException(
    _Inout_ void*, _In_ const void*, _In_ const void*) noexcept;

_STD_BEGIN

_EXPORT_STD class exception_ptr {
public:
    exception_ptr() noexcept {
        __ExceptionPtrCreate(this);
    }

    exception_ptr(nullptr_t) noexcept {
        __ExceptionPtrCreate(this);
    }

    ~exception_ptr() noexcept {
        __ExceptionPtrDestroy(this);
    }

    exception_ptr(const exception_ptr& _Rhs) noexcept {
        __ExceptionPtrCopy(this, &_Rhs);
    }

    exception_ptr& operator=(const exception_ptr& _Rhs) noexcept {
        __ExceptionPtrAssign(this, &_Rhs);
        return *this;
    }

    exception_ptr& operator=(nullptr_t) noexcept {
        exception_ptr _Ptr;
        __ExceptionPtrAssign(this, &_Ptr);
        return *this;
    }

    explicit operator bool() const noexcept {
        return __ExceptionPtrToBool(this);
    }

    static exception_ptr _Copy_exception(_In_ void* _Except, _In_ const void* _Ptr) {
        exception_ptr _Retval;
        if (!_Ptr) {
            // unsupported exceptions
            return _Retval;
        }
        __ExceptionPtrCopyException(&_Retval, _Except, _Ptr);
        return _Retval;
    }

    friend void swap(exception_ptr& _Lhs, exception_ptr& _Rhs) noexcept {
        __ExceptionPtrSwap(&_Lhs, &_Rhs);
    }

    _NODISCARD_FRIEND bool operator==(const exception_ptr& _Lhs, const exception_ptr& _Rhs) noexcept {
        return __ExceptionPtrCompare(&_Lhs, &_Rhs);
    }

    _NODISCARD_FRIEND bool operator==(const exception_ptr& _Lhs, nullptr_t) noexcept {
        return !_Lhs;
    }

#if !_HAS_CXX20
    _NODISCARD_FRIEND bool operator==(nullptr_t, const exception_ptr& _Rhs) noexcept {
        return !_Rhs;
    }

    _NODISCARD_FRIEND bool operator!=(const exception_ptr& _Lhs, const exception_ptr& _Rhs) noexcept {
        return !(_Lhs == _Rhs);
    }

    _NODISCARD_FRIEND bool operator!=(const exception_ptr& _Lhs, nullptr_t) noexcept {
        return !(_Lhs == nullptr);
    }

    _NODISCARD_FRIEND bool operator!=(nullptr_t, const exception_ptr& _Rhs) noexcept {
        return !(nullptr == _Rhs);
    }
#endif // !_HAS_CXX20

private:
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif // defined(__clang__)
    void* _Data1{};
    void* _Data2{};
#ifdef __clang__
#pragma clang diagnostic pop
#endif // defined(__clang__)
};

_EXPORT_STD _NODISCARD inline exception_ptr current_exception() noexcept {
    exception_ptr _Retval;
    __ExceptionPtrCurrentException(&_Retval);
    return _Retval;
}

_EXPORT_STD [[noreturn]] inline void rethrow_exception(_In_ exception_ptr _Ptr) {
    __ExceptionPtrRethrow(&_Ptr);
}

template <class _Ex>
void* __GetExceptionInfo(_Ex);

_EXPORT_STD template <class _Ex>
_NODISCARD_SMART_PTR_ALLOC exception_ptr make_exception_ptr(_Ex _Except) noexcept {
    return exception_ptr::_Copy_exception(_STD addressof(_Except), __GetExceptionInfo(_Except));
}

_EXPORT_STD class nested_exception { // wrap an exception_ptr
public:
    nested_exception() noexcept : _Exc(_STD current_exception()) {}

    nested_exception(const nested_exception&) noexcept            = default;
    nested_exception& operator=(const nested_exception&) noexcept = default;
    virtual ~nested_exception() noexcept {}

    [[noreturn]] void rethrow_nested() const { // throw wrapped exception_ptr
        if (_Exc) {
            _STD rethrow_exception(_Exc);
        } else {
            _STD terminate(); // per N4950 [except.nested]/4
        }
    }

    _NODISCARD exception_ptr nested_ptr() const noexcept { // return wrapped exception_ptr
        return _Exc;
    }

private:
    exception_ptr _Exc;
};

template <class _Uty>
struct _With_nested_v2 : _Uty, nested_exception { // glue user exception to nested_exception
    template <class _Ty>
    explicit _With_nested_v2(_Ty&& _Arg)
        : _Uty(_STD forward<_Ty>(_Arg)), nested_exception() {} // store user exception and current_exception()
};

_EXPORT_STD template <class _Ty>
[[noreturn]] void throw_with_nested(_Ty&& _Arg) {
    // throw user exception, glued to nested_exception if possible
    using _Uty = decay_t<_Ty>;

    if constexpr (is_class_v<_Uty> && !is_base_of_v<nested_exception, _Uty> && !is_final_v<_Uty>) {
        // throw user exception glued to nested_exception
        _THROW(_With_nested_v2<_Uty>(_STD forward<_Ty>(_Arg)));
    } else {
        // throw user exception by itself
        _THROW(_STD forward<_Ty>(_Arg));
    }
}

#ifdef _CPPRTTI
_EXPORT_STD template <class _Ty>
void rethrow_if_nested(const _Ty& _Arg) {
    // detect nested_exception inheritance
    constexpr bool _Can_use_dynamic_cast =
        is_polymorphic_v<_Ty> && (!is_base_of_v<nested_exception, _Ty> || is_convertible_v<_Ty*, nested_exception*>);

    if constexpr (_Can_use_dynamic_cast) {
        const auto _Nested = dynamic_cast<const nested_exception*>(_STD addressof(_Arg));

        if (_Nested) {
            _Nested->rethrow_nested();
        }
    }
}
#else // ^^^ defined(_CPPRTTI) / !defined(_CPPRTTI) vvv
_EXPORT_STD template <class _Ty>
void rethrow_if_nested(const _Ty&) = delete; // requires /GR option
#endif // ^^^ !defined(_CPPRTTI) ^^^

_EXPORT_STD class bad_variant_access
    : public exception { // exception for visit of a valueless variant or get<I> on a variant with index() != I
public:
    bad_variant_access() noexcept = default;

    _NODISCARD const char* __CLR_OR_THIS_CALL what() const noexcept override {
        return "bad variant access";
    }

#if !_HAS_EXCEPTIONS
protected:
    void _Doraise() const override { // perform class-specific exception handling
        _RAISE(*this);
    }
#endif // ^^^ !_HAS_EXCEPTIONS ^^^
};

[[noreturn]] inline void _Throw_bad_variant_access() {
    _THROW(bad_variant_access{});
}

_STD_END

#pragma pop_macro("new")
_STL_RESTORE_CLANG_WARNINGS
#pragma warning(pop)
#pragma pack(pop)

#endif // _STL_COMPILER_PREPROCESSOR
#endif // _EXCEPTION_
