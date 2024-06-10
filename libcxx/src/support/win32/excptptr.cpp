//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This implementation communicates with the EH runtime though vcruntime's per-thread-data structure; see
// _pCurrentException in <trnsctrl.h>.
//
// As a result, normal EH runtime services (such as noexcept functions) are safe to use in this file.

#ifndef _VCRT_ALLOW_INTERNALS
#define _VCRT_ALLOW_INTERNALS
#endif // !defined(_VCRT_ALLOW_INTERNALS)

#include <Unknwn.h>
#include <cstdlib> // for abort
#include <cstring> // for memcpy
#include <eh.h>
#include <ehdata.h>
#include <exception>
#include <internal_shared.h>
#include <malloc.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <trnsctrl.h>
#include <xcall_once.h>

#include <Windows.h>

// Pre-V4 managed exception code
#define MANAGED_EXCEPTION_CODE 0XE0434F4D

// V4 and later managed exception code
#define MANAGED_EXCEPTION_CODE_V4 0XE0434352

extern "C" _CRTIMP2 void* __cdecl __AdjustPointer(void*, const PMD&); // defined in frame.cpp

using namespace std;

namespace {
#if defined(_M_CEE_PURE)
    template <class _Ty>
    _Ty& _Immortalize() { // return a reference to an object that will live forever
        /* MAGIC */ static _Immortalizer_impl<_Ty> _Static;
        return reinterpret_cast<_Ty&>(_Static._Storage);
    }
#elif !defined(_M_CEE)
    template <class _Ty>
    struct _Constexpr_excptptr_immortalize_impl {
        union {
            _Ty _Storage;
        };

        constexpr _Constexpr_excptptr_immortalize_impl() noexcept : _Storage{} {}

        _Constexpr_excptptr_immortalize_impl(const _Constexpr_excptptr_immortalize_impl&)            = delete;
        _Constexpr_excptptr_immortalize_impl& operator=(const _Constexpr_excptptr_immortalize_impl&) = delete;

        _MSVC_NOOP_DTOR ~_Constexpr_excptptr_immortalize_impl() {
            // do nothing, allowing _Ty to be used during shutdown
        }
    };

    template <class _Ty>
    _Constexpr_excptptr_immortalize_impl<_Ty> _Immortalize_impl;

    template <class _Ty>
    [[nodiscard]] _Ty& _Immortalize() noexcept {
        return _Immortalize_impl<_Ty>._Storage;
    }
#else // ^^^ !defined(_M_CEE) / defined(_M_CEE), TRANSITION, VSO-1153256 vvv
    template <class _Ty>
    _Ty& _Immortalize() { // return a reference to an object that will live forever
        static once_flag _Flag;
        alignas(_Ty) static unsigned char _Storage[sizeof(_Ty)];
        call_once(_Flag, [&_Storage] { ::new (static_cast<void*>(&_Storage)) _Ty(); });
        return reinterpret_cast<_Ty&>(_Storage);
    }
#endif // ^^^ !defined(_M_CEE_PURE) && defined(_M_CEE), TRANSITION, VSO-1153256 ^^^

    void _PopulateCppExceptionRecord(
        _EXCEPTION_RECORD& _Record, const void* const _PExcept, ThrowInfo* _PThrow) noexcept {
        _Record.ExceptionCode           = EH_EXCEPTION_NUMBER;
        _Record.ExceptionFlags          = EXCEPTION_NONCONTINUABLE;
        _Record.ExceptionRecord         = nullptr; // no SEH to chain
        _Record.ExceptionAddress        = nullptr; // Address of exception. Will be overwritten by OS
        _Record.NumberParameters        = EH_EXCEPTION_PARAMETERS;
        _Record.ExceptionInformation[0] = EH_MAGIC_NUMBER1; // params.magicNumber
        _Record.ExceptionInformation[1] = reinterpret_cast<ULONG_PTR>(_PExcept); // params.pExceptionObject

        if (_PThrow && (_PThrow->attributes & TI_IsWinRT)) {
            // The pointer to the ExceptionInfo structure is stored sizeof(void*) in front of each WinRT Exception Info.
            const auto _PWei = (*static_cast<WINRTEXCEPTIONINFO** const*>(_PExcept))[-1];
            _PThrow          = _PWei->throwInfo;
        }

        _Record.ExceptionInformation[2] = reinterpret_cast<ULONG_PTR>(_PThrow); // params.pThrowInfo

#if _EH_RELATIVE_TYPEINFO
        void* _ThrowImageBase =
            _PThrow ? RtlPcToFileHeader(const_cast<void*>(static_cast<const void*>(_PThrow)), &_ThrowImageBase)
                    : nullptr;
        _Record.ExceptionInformation[3] = reinterpret_cast<ULONG_PTR>(_ThrowImageBase); // params.pThrowImageBase
#endif // _EH_RELATIVE_TYPEINFO

        // If the throw info indicates this throw is from a pure region,
        // set the magic number to the Pure one, so only a pure-region
        // catch will see it.
        //
        // Also use the Pure magic number on 64-bit platforms if we were unable to
        // determine an image base, since that was the old way to determine
        // a pure throw, before the TI_IsPure bit was added to the FuncInfo
        // attributes field.
        if (_PThrow
            && ((_PThrow->attributes & TI_IsPure)
#if _EH_RELATIVE_TYPEINFO
                || !_ThrowImageBase
#endif // _EH_RELATIVE_TYPEINFO
                )) {
            _Record.ExceptionInformation[0] = EH_PURE_MAGIC_NUMBER1;
        }
    }

    void _CopyExceptionRecord(_EXCEPTION_RECORD& _Dest, const _EXCEPTION_RECORD& _Src) noexcept {
        _Dest.ExceptionCode = _Src.ExceptionCode;
        // we force EXCEPTION_NONCONTINUABLE because rethrow_exception is [[noreturn]]
        _Dest.ExceptionFlags   = _Src.ExceptionFlags | EXCEPTION_NONCONTINUABLE;
        _Dest.ExceptionRecord  = nullptr; // We don't chain SEH exceptions
        _Dest.ExceptionAddress = nullptr; // Useless field to copy. It will be overwritten by RaiseException()
        const auto _Parameters = _Src.NumberParameters;
        _Dest.NumberParameters = _Parameters;

        // copy the number of parameters in use
        constexpr auto _Max_parameters = static_cast<DWORD>(EXCEPTION_MAXIMUM_PARAMETERS);
        const auto _In_use             = (_STD min)(_Parameters, _Max_parameters);
        _CSTD memcpy(_Dest.ExceptionInformation, _Src.ExceptionInformation, _In_use * sizeof(ULONG_PTR));
        _CSTD memset(&_Dest.ExceptionInformation[_In_use], 0, (_Max_parameters - _In_use) * sizeof(ULONG_PTR));
    }

    void _CopyExceptionObject(void* _Dest, const void* _Src, const CatchableType* const _PType
#if _EH_RELATIVE_TYPEINFO
        ,
        const uintptr_t _ThrowImageBase
#endif // _EH_RELATIVE_TYPEINFO
    ) {
        // copy an object of type denoted by *_PType from _Src to _Dest; throws whatever the copy ctor of the type
        // denoted by *_PType throws
        if ((_PType->properties & CT_IsSimpleType) || _PType->copyFunction == 0) {
            memcpy(_Dest, _Src, _PType->sizeOrOffset);

            if (_PType->properties & CT_IsWinRTHandle) {
                const auto _PUnknown = *static_cast<IUnknown* const*>(_Src);
                if (_PUnknown) {
                    _PUnknown->AddRef();
                }
            }
            return;
        }

#if _EH_RELATIVE_TYPEINFO
        const auto _CopyFunc = reinterpret_cast<void*>(_ThrowImageBase + _PType->copyFunction);
#else // ^^^ _EH_RELATIVE_TYPEINFO / !_EH_RELATIVE_TYPEINFO vvv
        const auto _CopyFunc = _PType->copyFunction;
#endif // ^^^ !_EH_RELATIVE_TYPEINFO ^^^

        const auto _Adjusted = __AdjustPointer(const_cast<void*>(_Src), _PType->thisDisplacement);
        if (_PType->properties & CT_HasVirtualBase) {
#ifdef _M_CEE_PURE
            reinterpret_cast<void(__clrcall*)(void*, void*, int)>(_CopyFunc)(_Dest, _Adjusted, 1);
#else // ^^^ defined(_M_CEE_PURE) / !defined(_M_CEE_PURE) vvv
            _CallMemberFunction2(_Dest, _CopyFunc, _Adjusted, 1);
#endif // ^^^ !defined(_M_CEE_PURE) ^^^
        } else {
#ifdef _M_CEE_PURE
            reinterpret_cast<void(__clrcall*)(void*, void*)>(_CopyFunc)(_Dest, _Adjusted);
#else // ^^^ defined(_M_CEE_PURE) / !defined(_M_CEE_PURE) vvv
            _CallMemberFunction1(_Dest, _CopyFunc, _Adjusted);
#endif // ^^^ !defined(_M_CEE_PURE) ^^^
        }
    }
} // unnamed namespace

// All exception_ptr implementations are out-of-line because <memory> depends on <exception>,
// which means <exception> cannot include <memory> -- and shared_ptr is defined in <memory>.
// To workaround this, we created a dummy class exception_ptr, which is structurally identical to shared_ptr.

_STD_BEGIN
struct _Exception_ptr_access {
    template <class _Ty, class _Ty2>
    static void _Set_ptr_rep(_Ptr_base<_Ty>& _This, _Ty2* _Px, _Ref_count_base* _Rx) noexcept {
        _This._Ptr = _Px;
        _This._Rep = _Rx;
    }
};
_STD_END

static_assert(sizeof(exception_ptr) == sizeof(shared_ptr<const _EXCEPTION_RECORD>)
                  && alignof(exception_ptr) == alignof(shared_ptr<const _EXCEPTION_RECORD>),
    "std::exception_ptr and std::shared_ptr<const _EXCEPTION_RECORD> must have the same layout.");

namespace {
    template <class _StaticEx>
    class _ExceptionPtr_static final : public _Ref_count_base {
        // reference count control block for special "never allocates" exceptions like the bad_alloc or bad_exception
        // exception_ptrs
    private:
        void _Destroy() noexcept override {
            // intentionally does nothing
        }

        void _Delete_this() noexcept override {
            // intentionally does nothing
        }

    public:
        // constexpr, TRANSITION P1064
        explicit _ExceptionPtr_static() noexcept : _Ref_count_base() {
            _PopulateCppExceptionRecord(_ExRecord, &_Ex, static_cast<ThrowInfo*>(__GetExceptionInfo(_Ex)));
        }

        static shared_ptr<const _EXCEPTION_RECORD> _Get() noexcept {
            auto& _Instance = _Immortalize<_ExceptionPtr_static>();
            shared_ptr<const _EXCEPTION_RECORD> _Ret;
            _Instance._Incref();
            _Exception_ptr_access::_Set_ptr_rep(_Ret, &_Instance._ExRecord, &_Instance);
            return _Ret;
        }

        _EXCEPTION_RECORD _ExRecord;
        _StaticEx _Ex;
    };

    class _ExceptionPtr_normal final : public _Ref_count_base {
        // reference count control block for exception_ptrs; the exception object is stored at
        // reinterpret_cast<unsigned char*>(this) + sizeof(_ExceptionPtr_normal)
    private:
        void _Destroy() noexcept override {
            // call the destructor for a stored pure or native C++ exception if necessary
            const auto& _CppEhRecord = reinterpret_cast<EHExceptionRecord&>(_ExRecord);

            if (!PER_IS_MSVC_PURE_OR_NATIVE_EH(&_CppEhRecord)) {
                return;
            }

            const auto _PThrow = _CppEhRecord.params.pThrowInfo;
            if (!_PThrow) {
                // No ThrowInfo exists. If this was a C++ exception, something must have corrupted it.
                _CSTD abort();
            }

            if (!_CppEhRecord.params.pExceptionObject) {
                return;
            }

#if _EH_RELATIVE_TYPEINFO
            const auto _ThrowImageBase     = reinterpret_cast<uintptr_t>(_CppEhRecord.params.pThrowImageBase);
            const auto _CatchableTypeArray = reinterpret_cast<const CatchableTypeArray*>(
                static_cast<uintptr_t>(_PThrow->pCatchableTypeArray) + _ThrowImageBase);
            const auto _PType = reinterpret_cast<CatchableType*>(
                static_cast<uintptr_t>(_CatchableTypeArray->arrayOfCatchableTypes[0]) + _ThrowImageBase);
#else // ^^^ _EH_RELATIVE_TYPEINFO / !_EH_RELATIVE_TYPEINFO vvv
            const auto _PType = _PThrow->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // ^^^ !_EH_RELATIVE_TYPEINFO ^^^

            if (_PThrow->pmfnUnwind) {
                // The exception was a user defined type with a nontrivial destructor, call it
#if defined(_M_CEE_PURE)
                reinterpret_cast<void(__clrcall*)(void*)>(_PThrow->pmfnUnwind)(_CppEhRecord.params.pExceptionObject);
#elif _EH_RELATIVE_TYPEINFO
                _CallMemberFunction0(_CppEhRecord.params.pExceptionObject,
                    reinterpret_cast<void*>(_PThrow->pmfnUnwind + _ThrowImageBase));
#else // ^^^ _EH_RELATIVE_TYPEINFO && !defined(_M_CEE_PURE) / !_EH_RELATIVE_TYPEINFO && !defined(_M_CEE_PURE) vvv
                _CallMemberFunction0(_CppEhRecord.params.pExceptionObject, _PThrow->pmfnUnwind);
#endif // ^^^ !_EH_RELATIVE_TYPEINFO && !defined(_M_CEE_PURE) ^^^
            } else if (_PType->properties & CT_IsWinRTHandle) {
                const auto _PUnknown = *static_cast<IUnknown* const*>(_CppEhRecord.params.pExceptionObject);
                if (_PUnknown) {
                    _PUnknown->Release();
                }
            }
        }

        void _Delete_this() noexcept override {
            free(this);
        }

    public:
        explicit _ExceptionPtr_normal(const _EXCEPTION_RECORD& _Record) noexcept : _Ref_count_base() {
            _CopyExceptionRecord(_ExRecord, _Record);
        }

        _EXCEPTION_RECORD _ExRecord;
        void* _Unused_alignment_padding{};
    };

    // We aren't using alignas because this file might be compiled with _M_CEE_PURE
    static_assert(sizeof(_ExceptionPtr_normal) % __STDCPP_DEFAULT_NEW_ALIGNMENT__ == 0,
        "Exception in exception_ptr would be constructed with the wrong alignment");

    void _Assign_seh_exception_ptr_from_record(
        shared_ptr<const _EXCEPTION_RECORD>& _Dest, const _EXCEPTION_RECORD& _Record, void* const _RxRaw) noexcept {
        // in the memory _RxRaw, constructs a reference count control block for a SEH exception denoted by _Record
        // if _RxRaw is nullptr, assigns bad_alloc instead
        if (!_RxRaw) {
            _Dest = _ExceptionPtr_static<bad_alloc>::_Get();
            return;
        }

        const auto _Rx = ::new (_RxRaw) _ExceptionPtr_normal(_Record);
        _Exception_ptr_access::_Set_ptr_rep(_Dest, &_Rx->_ExRecord, _Rx);
    }

    void _Assign_cpp_exception_ptr_from_record(
        shared_ptr<const _EXCEPTION_RECORD>& _Dest, const EHExceptionRecord& _Record) noexcept {
        // construct a reference count control block for the C++ exception recorded by _Record, and bind it to _Dest
        // if allocating memory for the reference count control block fails, sets _Dest to bad_alloc
        // if copying the exception object referred to _Record throws, constructs a reference count control block for
        //      that exception instead
        // if copying the exception object thrown by copying the original exception object throws, sets _Dest to
        //      bad_exception
        const auto _PThrow = _Record.params.pThrowInfo;
#if _EH_RELATIVE_TYPEINFO
        const auto _ThrowImageBase     = reinterpret_cast<uintptr_t>(_Record.params.pThrowImageBase);
        const auto _CatchableTypeArray = reinterpret_cast<const CatchableTypeArray*>(
            static_cast<uintptr_t>(_PThrow->pCatchableTypeArray) + _ThrowImageBase);
        const auto _PType = reinterpret_cast<CatchableType*>(
            static_cast<uintptr_t>(_CatchableTypeArray->arrayOfCatchableTypes[0]) + _ThrowImageBase);
#else // ^^^ _EH_RELATIVE_TYPEINFO / !_EH_RELATIVE_TYPEINFO vvv
        const auto _PType = _PThrow->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // ^^^ !_EH_RELATIVE_TYPEINFO ^^^

        const auto _ExceptionObjectSize = static_cast<size_t>(_PType->sizeOrOffset);
        const auto _AllocSize           = sizeof(_ExceptionPtr_normal) + _ExceptionObjectSize;
        _Analysis_assume_(_AllocSize >= sizeof(_ExceptionPtr_normal));
        auto _RxRaw = malloc(_AllocSize);
        if (!_RxRaw) {
            _Dest = _ExceptionPtr_static<bad_alloc>::_Get();
            return;
        }

        try {
            _CopyExceptionObject(static_cast<_ExceptionPtr_normal*>(_RxRaw) + 1, _Record.params.pExceptionObject, _PType
#if _EH_RELATIVE_TYPEINFO
                ,
                _ThrowImageBase
#endif // _EH_RELATIVE_TYPEINFO
            );

            const auto _Rx = ::new (_RxRaw) _ExceptionPtr_normal(reinterpret_cast<const _EXCEPTION_RECORD&>(_Record));
            reinterpret_cast<EHExceptionRecord&>(_Rx->_ExRecord).params.pExceptionObject =
                static_cast<_ExceptionPtr_normal*>(_RxRaw) + 1;
            _Exception_ptr_access::_Set_ptr_rep(_Dest, &_Rx->_ExRecord, _Rx);
        } catch (...) { // copying the exception object threw an exception
            const auto& _InnerRecord = *_pCurrentException; // exception thrown by the original exception's copy ctor
            if (_InnerRecord.ExceptionCode == MANAGED_EXCEPTION_CODE
                || _InnerRecord.ExceptionCode == MANAGED_EXCEPTION_CODE_V4) {
                // we don't support managed exceptions and don't want to say there's no active exception, so give up and
                // say bad_exception
                free(_RxRaw);
                _Dest = _ExceptionPtr_static<bad_exception>::_Get();
                return;
            }

            if (!PER_IS_MSVC_PURE_OR_NATIVE_EH(&_InnerRecord)) { // catching a non-C++ exception depends on /EHa
                _Assign_seh_exception_ptr_from_record(
                    _Dest, reinterpret_cast<const _EXCEPTION_RECORD&>(_InnerRecord), _RxRaw);
                return;
            }

            const auto _PInnerThrow = _InnerRecord.params.pThrowInfo;
#if _EH_RELATIVE_TYPEINFO
            const auto _InnerThrowImageBase     = reinterpret_cast<uintptr_t>(_InnerRecord.params.pThrowImageBase);
            const auto _InnerCatchableTypeArray = reinterpret_cast<const CatchableTypeArray*>(
                static_cast<uintptr_t>(_PInnerThrow->pCatchableTypeArray) + _InnerThrowImageBase);
            const auto _PInnerType = reinterpret_cast<CatchableType*>(
                static_cast<uintptr_t>(_InnerCatchableTypeArray->arrayOfCatchableTypes[0]) + _InnerThrowImageBase);
#else // ^^^ _EH_RELATIVE_TYPEINFO / !_EH_RELATIVE_TYPEINFO vvv
            const auto _PInnerType = _PInnerThrow->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // ^^^ !_EH_RELATIVE_TYPEINFO ^^^

            const auto _InnerExceptionSize = static_cast<size_t>(_PInnerType->sizeOrOffset);
            const auto _InnerAllocSize     = sizeof(_ExceptionPtr_normal) + _InnerExceptionSize;
            if (_InnerAllocSize > _AllocSize) {
                free(_RxRaw);
                _RxRaw = malloc(_InnerAllocSize);
                if (!_RxRaw) {
                    _Dest = _ExceptionPtr_static<bad_alloc>::_Get();
                    return;
                }
            }

            try {
                _CopyExceptionObject(
                    static_cast<_ExceptionPtr_normal*>(_RxRaw) + 1, _InnerRecord.params.pExceptionObject, _PInnerType
#if _EH_RELATIVE_TYPEINFO
                    ,
                    _ThrowImageBase
#endif // _EH_RELATIVE_TYPEINFO
                );
            } catch (...) { // copying the exception emitted while copying the original exception also threw, give up
                free(_RxRaw);
                _Dest = _ExceptionPtr_static<bad_exception>::_Get();
                return;
            }

            // this next block must be duplicated inside the catch (even though it looks identical to the block in the
            // try) so that _InnerRecord is held alive; exiting the catch will destroy it
            const auto _Rx =
                ::new (_RxRaw) _ExceptionPtr_normal(reinterpret_cast<const _EXCEPTION_RECORD&>(_InnerRecord));
            reinterpret_cast<EHExceptionRecord&>(_Rx->_ExRecord).params.pExceptionObject =
                static_cast<_ExceptionPtr_normal*>(_RxRaw) + 1;
            _Exception_ptr_access::_Set_ptr_rep(_Dest, &_Rx->_ExRecord, _Rx);
        }
    }
} // unnamed namespace

_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCreate(_Out_ void* _Ptr) noexcept {
    ::new (_Ptr) shared_ptr<const _EXCEPTION_RECORD>();
}

_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrDestroy(_Inout_ void* _Ptr) noexcept {
    static_cast<shared_ptr<const _EXCEPTION_RECORD>*>(_Ptr)->~shared_ptr<const _EXCEPTION_RECORD>();
}

_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCopy(_Out_ void* _Dest, _In_ const void* _Src) noexcept {
    ::new (_Dest) shared_ptr<const _EXCEPTION_RECORD>(*static_cast<const shared_ptr<const _EXCEPTION_RECORD>*>(_Src));
}

_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrAssign(_Inout_ void* _Dest, _In_ const void* _Src) noexcept {
    *static_cast<shared_ptr<const _EXCEPTION_RECORD>*>(_Dest) =
        *static_cast<const shared_ptr<const _EXCEPTION_RECORD>*>(_Src);
}

_CRTIMP2_PURE bool __CLRCALL_PURE_OR_CDECL __ExceptionPtrCompare(
    _In_ const void* _Lhs, _In_ const void* _Rhs) noexcept {
    return *static_cast<const shared_ptr<const _EXCEPTION_RECORD>*>(_Lhs)
        == *static_cast<const shared_ptr<const _EXCEPTION_RECORD>*>(_Rhs);
}

_CRTIMP2_PURE bool __CLRCALL_PURE_OR_CDECL __ExceptionPtrToBool(_In_ const void* _Ptr) noexcept {
    return static_cast<bool>(*static_cast<const shared_ptr<const _EXCEPTION_RECORD>*>(_Ptr));
}

_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrSwap(_Inout_ void* _Lhs, _Inout_ void* _Rhs) noexcept {
    static_cast<shared_ptr<const _EXCEPTION_RECORD>*>(_Lhs)->swap(
        *static_cast<shared_ptr<const _EXCEPTION_RECORD>*>(_Rhs));
}

_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCurrentException(void* _Ptr) noexcept {
    const auto _PRecord = _pCurrentException; // nontrivial FLS cost, pay it once
    if (!_PRecord || _PRecord->ExceptionCode == MANAGED_EXCEPTION_CODE
        || _PRecord->ExceptionCode == MANAGED_EXCEPTION_CODE_V4) {
        return; // no current exception, or we don't support managed exceptions
    }

    auto& _Dest = *static_cast<shared_ptr<const _EXCEPTION_RECORD>*>(_Ptr);
    if (PER_IS_MSVC_PURE_OR_NATIVE_EH(_PRecord)) {
        _Assign_cpp_exception_ptr_from_record(_Dest, *_PRecord);
    } else {
        // _Assign_seh_exception_ptr_from_record handles failed malloc
        _Assign_seh_exception_ptr_from_record(
            _Dest, reinterpret_cast<_EXCEPTION_RECORD&>(*_PRecord), malloc(sizeof(_ExceptionPtr_normal)));
    }
}

[[noreturn]] _CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrRethrow(_In_ const void* _PtrRaw) {
    const shared_ptr<const _EXCEPTION_RECORD>* _Ptr = static_cast<const shared_ptr<const _EXCEPTION_RECORD>*>(_PtrRaw);
    // throwing a bad_exception if they give us a nullptr exception_ptr
    if (!*_Ptr) {
        throw bad_exception();
    }

    auto _RecordCopy = **_Ptr;
    auto& _CppRecord = reinterpret_cast<EHExceptionRecord&>(_RecordCopy);
    if (PER_IS_MSVC_PURE_OR_NATIVE_EH(&_CppRecord)) {
        // This is a C++ exception.
        // We need to build the exception on the stack because the current exception mechanism assumes the exception
        // object is on the stack and will call the appropriate destructor (if there's a nontrivial one).
        const auto _PThrow = _CppRecord.params.pThrowInfo;
        if (!_CppRecord.params.pExceptionObject || !_PThrow || !_PThrow->pCatchableTypeArray) {
            // Missing or corrupt ThrowInfo. If this was a C++ exception, something must have corrupted it.
            _CSTD abort();
        }

#if _EH_RELATIVE_TYPEINFO
        const auto _ThrowImageBase = reinterpret_cast<uintptr_t>(_CppRecord.params.pThrowImageBase);
        const auto _CatchableTypeArray =
            reinterpret_cast<const CatchableTypeArray*>(_ThrowImageBase + _PThrow->pCatchableTypeArray);
#else // ^^^ _EH_RELATIVE_TYPEINFO / !_EH_RELATIVE_TYPEINFO vvv
        const auto _CatchableTypeArray = _PThrow->pCatchableTypeArray;
#endif // ^^^ !_EH_RELATIVE_TYPEINFO ^^^

        if (_CatchableTypeArray->nCatchableTypes <= 0) {
            // Ditto corrupted.
            _CSTD abort();
        }

        // we finally got the type info we want
#if _EH_RELATIVE_TYPEINFO
        const auto _PType = reinterpret_cast<CatchableType*>(
            static_cast<uintptr_t>(_CatchableTypeArray->arrayOfCatchableTypes[0]) + _ThrowImageBase);
#else // ^^^ _EH_RELATIVE_TYPEINFO / !_EH_RELATIVE_TYPEINFO vvv
        const auto _PType = _PThrow->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // ^^^ !_EH_RELATIVE_TYPEINFO ^^^

        // Alloc memory on stack for exception object. This might cause a stack overflow SEH exception, or another C++
        // exception when copying the C++ exception object. In that case, we just let that become the thrown exception.

#pragma warning(suppress : 6255) //  _alloca indicates failure by raising a stack overflow exception
        void* _PExceptionBuffer = alloca(_PType->sizeOrOffset);
        _CopyExceptionObject(_PExceptionBuffer, _CppRecord.params.pExceptionObject, _PType
#if _EH_RELATIVE_TYPEINFO
            ,
            _ThrowImageBase
#endif // _EH_RELATIVE_TYPEINFO
        );

        _CppRecord.params.pExceptionObject = _PExceptionBuffer;
    } else {
        // this is a SEH exception, no special handling is required
    }

    _Analysis_assume_(_RecordCopy.NumberParameters <= EXCEPTION_MAXIMUM_PARAMETERS);
    RaiseException(_RecordCopy.ExceptionCode, _RecordCopy.ExceptionFlags, _RecordCopy.NumberParameters,
        _RecordCopy.ExceptionInformation);
}

_CRTIMP2_PURE void __CLRCALL_PURE_OR_CDECL __ExceptionPtrCopyException(
    _Inout_ void* _Ptr, _In_ const void* _PExceptRaw, _In_ const void* _PThrowRaw) noexcept {
    _EXCEPTION_RECORD _Record;
    _PopulateCppExceptionRecord(_Record, _PExceptRaw, static_cast<ThrowInfo*>(_PThrowRaw));
    _Assign_cpp_exception_ptr_from_record(
        *static_cast<shared_ptr<const _EXCEPTION_RECORD>*>(_Ptr), reinterpret_cast<const EHExceptionRecord&>(_Record));
}
