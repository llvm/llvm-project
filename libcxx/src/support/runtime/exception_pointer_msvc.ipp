// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wmultichar")

#include <__exception/exception_ptr.h>
#include <__memory/shared_count.h>
#include <cstdlib>
#include <cstring>

#include <Unknwn.h>
#include <Windows.h>
struct _ThrowInfo;
#include <eh.h>
#include <ehdata.h>
#include <malloc.h> // alloca

// Pre-V4 managed exception code
#define MANAGED_EXCEPTION_CODE 0XE0434F4D

// V4 and later managed exception code
#define MANAGED_EXCEPTION_CODE_V4 0XE0434352

extern "C" _LIBCPP_CRT_FUNC void* __cdecl __AdjustPointer(void*, const PMD&);
extern "C" _LIBCPP_CRT_FUNC void** __cdecl __current_exception();

namespace {

typedef void(__stdcall* __prepare_for_throw_t)(void*);

struct __winrt_exception_info {
  void* __description;
  void* __restricted_error_string;
  void* __restricted_error_reference;
  void* __capability_sid;
  long __hr;
  void* __restricted_info;
  ThrowInfo* __throw_info;
  unsigned int __size;
  __prepare_for_throw_t __prepare_throw;
};

inline EHExceptionRecord* __get_current_exception() noexcept {
  return *reinterpret_cast<EHExceptionRecord**>(__current_exception());
}

inline void __call_member_function_0(void* __this, void* __mfn) {
  auto __fn = reinterpret_cast<void(__thiscall*)(void*)>(__mfn);
  __fn(__this);
}

inline void __call_member_function_1(void* __this, void* __mfn, void* __arg) {
  auto __fn = reinterpret_cast<void(__thiscall*)(void*, void*)>(__mfn);
  __fn(__this, __arg);
}

inline void __call_member_function_2(void* __this, void* __mfn, void* __arg1, int __arg2) {
  auto __fn = reinterpret_cast<void(__thiscall*)(void*, void*, int)>(__mfn);
  __fn(__this, __arg1, __arg2);
}

void __populate_cpp_exception_record(
    _EXCEPTION_RECORD& __record, const void* const __except_obj, ThrowInfo* __throw_info) noexcept {
  __record.ExceptionCode           = EH_EXCEPTION_NUMBER;
  __record.ExceptionFlags          = EXCEPTION_NONCONTINUABLE;
  __record.ExceptionRecord         = nullptr; // No SEH to chain.
  __record.ExceptionAddress        = nullptr; // Address of exception. This will be overwritten by OS.
  __record.NumberParameters        = EH_EXCEPTION_PARAMETERS;
  __record.ExceptionInformation[0] = EH_MAGIC_NUMBER1;
  __record.ExceptionInformation[1] = reinterpret_cast<ULONG_PTR>(__except_obj);

  if (__throw_info && (__throw_info->attributes & TI_IsWinRT)) {
    // The pointer to the ExceptionInfo structure is stored sizeof(void*) in front of each WinRT Exception Info.
    const auto __wei = (*static_cast<__winrt_exception_info** const*>(const_cast<void*>(__except_obj)))[-1];
    __throw_info     = __wei->__throw_info;
  }

  __record.ExceptionInformation[2] = reinterpret_cast<ULONG_PTR>(__throw_info);

#if _EH_RELATIVE_TYPEINFO
  void* __throw_image_base =
      __throw_info ? RtlPcToFileHeader(const_cast<void*>(static_cast<const void*>(__throw_info)), &__throw_image_base)
                   : nullptr;
  __record.ExceptionInformation[3] = reinterpret_cast<ULONG_PTR>(__throw_image_base);
#endif // _EH_RELATIVE_TYPEINFO

  // If the throw info indicates this throw is from a pure region, set the magic number to the Pure one, so only a
  // pure-region catch will see it.
  //
  // Also use the Pure magic number on 64-bit platforms if we were unable to determine an image base, since that was the
  // old way to determine a pure throw, before the TI_IsPure bit was added to the FuncInfo attributes field.
  if (__throw_info && ((__throw_info->attributes & TI_IsPure)
#if _EH_RELATIVE_TYPEINFO
                       || !__throw_image_base
#endif // _EH_RELATIVE_TYPEINFO
                       ))
    __record.ExceptionInformation[0] = EH_PURE_MAGIC_NUMBER1;
}

void __copy_exception_record(_EXCEPTION_RECORD& __dest, const _EXCEPTION_RECORD& __src) noexcept {
  __dest.ExceptionCode    = __src.ExceptionCode;
  // We force EXCEPTION_NONCONTINUABLE because rethrow_exception is [[noreturn]].
  __dest.ExceptionFlags   = __src.ExceptionFlags | EXCEPTION_NONCONTINUABLE;
  __dest.ExceptionRecord  = nullptr; // We don't chain SEH exceptions.
  __dest.ExceptionAddress = nullptr; // Useless field to copy. It will be overwritten by RaiseException().
  const auto __parameters = __src.NumberParameters;
  __dest.NumberParameters = __parameters;

  // Copy the number of parameters in use.
  constexpr auto __max_parameters = static_cast<DWORD>(EXCEPTION_MAXIMUM_PARAMETERS);
  const auto __in_use             = (__parameters < __max_parameters) ? __parameters : __max_parameters;
  std::memcpy(__dest.ExceptionInformation, __src.ExceptionInformation, __in_use * sizeof(ULONG_PTR));
  std::memset(&__dest.ExceptionInformation[__in_use], 0, (__max_parameters - __in_use) * sizeof(ULONG_PTR));
}

void __copy_exception_object(void* __dest, const void* __src, const CatchableType* const __type
#if _EH_RELATIVE_TYPEINFO
                             ,
                             uintptr_t __throw_image_base
#endif // _EH_RELATIVE_TYPEINFO
) {
  // Copy an object of type denoted by *_PType from __src to __dDest; throws whatever the copy ctor of the type denoted
  // by *_PType throws.
  if ((__type->properties & CT_IsSimpleType) || __type->copyFunction == 0) {
    std::memcpy(__dest, __src, __type->sizeOrOffset);

    if (__type->properties & CT_IsWinRTHandle) {
      const auto __unknown = *static_cast<IUnknown* const*>(const_cast<void*>(__src));
      if (__unknown)
        __unknown->AddRef();
    }
    return;
  }

#if _EH_RELATIVE_TYPEINFO
  const auto __copy_func = reinterpret_cast<void*>(__throw_image_base + __type->copyFunction);
#else // _EH_RELATIVE_TYPEINFO
  const auto __copy_func = __type->copyFunction;
#endif // _EH_RELATIVE_TYPEINFO

  const auto __adjusted = __AdjustPointer(const_cast<void*>(__src), __type->thisDisplacement);
  if (__type->properties & CT_HasVirtualBase)
    __call_member_function_2(__dest, __copy_func, __adjusted, 1);
  else
    __call_member_function_1(__dest, __copy_func, __adjusted);
}

struct __exception_ptr_storage : public std::__shared_count {
  _EXCEPTION_RECORD __record_;
  explicit __exception_ptr_storage(long __refs = 0) noexcept : std::__shared_count(__refs) {}
};

template <class _StaticEx>
struct __exception_ptr_static final : public __exception_ptr_storage {
  _StaticEx __ex_;

  __exception_ptr_static() noexcept : __exception_ptr_storage(0) {
    __populate_cpp_exception_record(__record_, &__ex_, static_cast<ThrowInfo*>(__GetExceptionInfo(__ex_)));
  }

  void __on_zero_shared() noexcept override {}

  static __exception_ptr_storage* __get() noexcept {
    struct __container {
      union {
        __exception_ptr_static __instance_;
      };
      __container() noexcept : __instance_() {}
      ~__container() {}
    };
    static __container __storage;
    return &__storage.__instance_;
  }
};

struct alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__) __exception_ptr_normal final : public __exception_ptr_storage {
  explicit __exception_ptr_normal(const _EXCEPTION_RECORD& __record) noexcept : __exception_ptr_storage(0) {
    __copy_exception_record(__record_, __record);
  }

  void __on_zero_shared() noexcept override {
    const auto& __cpp_eh_record = reinterpret_cast<EHExceptionRecord&>(__record_);
    if (PER_IS_MSVC_PURE_OR_NATIVE_EH(&__cpp_eh_record)) {
      const auto* __throw_info = __cpp_eh_record.params.pThrowInfo;
      if (__throw_info && __cpp_eh_record.params.pExceptionObject) {
#if _EH_RELATIVE_TYPEINFO
        const auto __throw_image_base = reinterpret_cast<uintptr_t>(__cpp_eh_record.params.pThrowImageBase);
        const auto* __catchable_type_array = reinterpret_cast<const CatchableTypeArray*>(
            static_cast<uintptr_t>(__throw_info->pCatchableTypeArray) + __throw_image_base);
        const auto* __type = reinterpret_cast<CatchableType*>(
            static_cast<uintptr_t>(__catchable_type_array->arrayOfCatchableTypes[0]) + __throw_image_base);
#else // _EH_RELATIVE_TYPEINFO
        const auto* __type = __throw_info->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // _EH_RELATIVE_TYPEINFO
        if (__throw_info->pmfnUnwind) {
          // The exception was a user defined type with a nontrivial destructor, call it.
#if _EH_RELATIVE_TYPEINFO
          __call_member_function_0(__cpp_eh_record.params.pExceptionObject,
                                   reinterpret_cast<void*>(__throw_info->pmfnUnwind + __throw_image_base));
#else // _EH_RELATIVE_TYPEINFO
          __call_member_function_0(__cpp_eh_record.params.pExceptionObject, __throw_info->pmfnUnwind);
#endif // _EH_RELATIVE_TYPEINFO
        } else if (__type->properties & CT_IsWinRTHandle) {
          const auto* __unknown = *static_cast<IUnknown* const*>(__cpp_eh_record.params.pExceptionObject);
          if (__unknown)
            const_cast<IUnknown*>(__unknown)->Release();
        }
      }
    }
    std::free(this);
  }
};

static_assert(sizeof(__exception_ptr_normal) % __STDCPP_DEFAULT_NEW_ALIGNMENT__ == 0,
              "Exception in exception_ptr would be constructed with the wrong alignment");

void __assign_seh_exception_ptr_from_record(
    void*& __dest, const _EXCEPTION_RECORD& __record, void* const __rx_raw) noexcept {
  if (!__rx_raw) {
    __dest = __exception_ptr_static<std::bad_alloc>::__get();
    return;
  }

  __dest = ::new (__rx_raw) __exception_ptr_normal(__record);
}

void __assign_cpp_exception_ptr_from_record(void*& __dest, const EHExceptionRecord& __record) noexcept {
  // Construct a reference count control block for the C++ exception recorded by __record, and bind it to __dest.
  //
  // * If allocating memory for the reference count control block fails, sets __dest to bad_alloc.
  // * If copying the exception object referred to __record throws, constructs a reference count control block for that
  //   exception instead.
  // * If copying the exception object thrown by copying the original exception object throws, sets __dest to
  //   bad_exception.
  const auto* __throw_info = __record.params.pThrowInfo;
#if _EH_RELATIVE_TYPEINFO
  const auto __throw_image_base = reinterpret_cast<uintptr_t>(__record.params.pThrowImageBase);
  const auto* __catchable_type_array = reinterpret_cast<const CatchableTypeArray*>(
      static_cast<uintptr_t>(__throw_info->pCatchableTypeArray) + __throw_image_base);
  const auto* __type = reinterpret_cast<CatchableType*>(
      static_cast<uintptr_t>(__catchable_type_array->arrayOfCatchableTypes[0]) + __throw_image_base);
#else // _EH_RELATIVE_TYPEINFO
  const auto* __type = __throw_info->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // _EH_RELATIVE_TYPEINFO

  const auto __except_obj_size = static_cast<size_t>(__type->sizeOrOffset);
  const auto __alloc_size      = sizeof(__exception_ptr_normal) + __except_obj_size;
  auto* __rx_raw               = std::malloc(__alloc_size);
  if (!__rx_raw) {
    __dest = __exception_ptr_static<std::bad_alloc>::__get();
    return;
  }

  try {
    __copy_exception_object(static_cast<__exception_ptr_normal*>(__rx_raw) + 1,
                            __record.params.pExceptionObject,
                            __type
#if _EH_RELATIVE_TYPEINFO
                            ,
                            __throw_image_base
#endif // _EH_RELATIVE_TYPEINFO
    );

    const auto* __rx = ::new (__rx_raw) __exception_ptr_normal(reinterpret_cast<const _EXCEPTION_RECORD&>(__record));
    reinterpret_cast<EHExceptionRecord&>(const_cast<__exception_ptr_normal*>(__rx)->__record_).params.pExceptionObject =
        static_cast<__exception_ptr_normal*>(__rx_raw) + 1;
    __dest = const_cast<__exception_ptr_normal*>(__rx);
  } catch (...) { // Copying the exception object threw an exception.
    // Exception thrown by the original exception's copy ctor.
    const auto* __inner_record_ptr = __get_current_exception();
    if (!__inner_record_ptr) {
      std::free(__rx_raw);
      __dest = __exception_ptr_static<std::bad_exception>::__get();
      return;
    }
    const auto& __inner_record = *__inner_record_ptr;
    if (__inner_record.ExceptionCode == MANAGED_EXCEPTION_CODE ||
        __inner_record.ExceptionCode == MANAGED_EXCEPTION_CODE_V4) {
      // We don't support managed exceptions and don't want to say there's no active exception, so give up and say
      // bad_exception.
      std::free(__rx_raw);
      __dest = __exception_ptr_static<std::bad_exception>::__get();
      return;
    }

    if (!PER_IS_MSVC_PURE_OR_NATIVE_EH(&__inner_record)) { // Catching a non-C++ exception depends on /EHa.
      __assign_seh_exception_ptr_from_record(
          __dest, reinterpret_cast<const _EXCEPTION_RECORD&>(__inner_record), __rx_raw);
      return;
    }

    const auto* __inner_throw = __inner_record.params.pThrowInfo;
#if _EH_RELATIVE_TYPEINFO
    const auto __inner_throw_image_base = reinterpret_cast<uintptr_t>(__inner_record.params.pThrowImageBase);
    const auto* __inner_catchable_type_array = reinterpret_cast<const CatchableTypeArray*>(
        static_cast<uintptr_t>(__inner_throw->pCatchableTypeArray) + __inner_throw_image_base);
    const auto* __inner_type = reinterpret_cast<CatchableType*>(
        static_cast<uintptr_t>(__inner_catchable_type_array->arrayOfCatchableTypes[0]) + __inner_throw_image_base);
#else // _EH_RELATIVE_TYPEINFO
    const auto* __inner_type = __inner_throw->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // _EH_RELATIVE_TYPEINFO

    const auto __inner_except_size = static_cast<size_t>(__inner_type->sizeOrOffset);
    const auto __inner_alloc_size  = sizeof(__exception_ptr_normal) + __inner_except_size;
    if (__inner_alloc_size > __alloc_size) {
      std::free(__rx_raw);
      __rx_raw = std::malloc(__inner_alloc_size);
      if (!__rx_raw) {
        __dest = __exception_ptr_static<std::bad_alloc>::__get();
        return;
      }
    }

    try {
      __copy_exception_object(static_cast<__exception_ptr_normal*>(__rx_raw) + 1,
                              __inner_record.params.pExceptionObject,
                              __inner_type
#if _EH_RELATIVE_TYPEINFO
                              ,
                              __inner_throw_image_base
#endif // _EH_RELATIVE_TYPEINFO
      );
    } catch (...) { // Copying the exception emitted while copying the original exception also threw, give up.
      std::free(__rx_raw);
      __dest = __exception_ptr_static<std::bad_exception>::__get();
      return;
    }

    // This next block must be duplicated inside the catch (even though it looks identical to the block in the try) so
    // that __inner_record is held alive; exiting the catch will destroy it.
    const auto* __rx =
        ::new (__rx_raw) __exception_ptr_normal(reinterpret_cast<const _EXCEPTION_RECORD&>(__inner_record));
    reinterpret_cast<EHExceptionRecord&>(const_cast<__exception_ptr_normal*>(__rx)->__record_).params.pExceptionObject =
        static_cast<__exception_ptr_normal*>(__rx_raw) + 1;
    __dest = const_cast<__exception_ptr_normal*>(__rx);
  }
}

} // namespace

namespace std {

exception_ptr::~exception_ptr() noexcept {
  if (__ptr_)
    static_cast<__shared_count*>(__ptr_)->__release_shared();
}

exception_ptr::exception_ptr(const exception_ptr& __other) noexcept : __ptr_(__other.__ptr_) {
  if (__ptr_)
    static_cast<__shared_count*>(__ptr_)->__add_shared();
}

exception_ptr& exception_ptr::operator=(const exception_ptr& __other) noexcept {
  if (__ptr_ != __other.__ptr_) {
    if (__other.__ptr_)
      static_cast<__shared_count*>(__other.__ptr_)->__add_shared();
    if (__ptr_)
      static_cast<__shared_count*>(__ptr_)->__release_shared();
    __ptr_ = __other.__ptr_;
  }
  return *this;
}

exception_ptr exception_ptr::__from_native_exception_pointer(void* __p) noexcept {
  exception_ptr __ret;
  __ret.__ptr_ = __p;
  if (__ret.__ptr_)
    static_cast<__shared_count*>(__ret.__ptr_)->__add_shared();
  return __ret;
}

exception_ptr __copy_exception_ptr(void* __except, const void* __ptr) {
  exception_ptr __ret = nullptr;
  if (!__ptr)
    return __ret;

  _EXCEPTION_RECORD __record;
  __populate_cpp_exception_record(__record, __except, static_cast<ThrowInfo*>(const_cast<void*>(__ptr)));
  __assign_cpp_exception_ptr_from_record(__ret.__ptr_, reinterpret_cast<const EHExceptionRecord&>(__record));
  return __ret;
}

exception_ptr current_exception() noexcept {
  exception_ptr __ret;
  const auto* __record = __get_current_exception();
  if (!__record || __record->ExceptionCode == MANAGED_EXCEPTION_CODE ||
      __record->ExceptionCode == MANAGED_EXCEPTION_CODE_V4)
    return __ret;

  if (PER_IS_MSVC_PURE_OR_NATIVE_EH(__record))
    __assign_cpp_exception_ptr_from_record(__ret.__ptr_, *__record);
  else
    __assign_seh_exception_ptr_from_record(
        __ret.__ptr_, reinterpret_cast<const _EXCEPTION_RECORD&>(*__record), std::malloc(sizeof(__exception_ptr_normal)));
  return __ret;
}

[[noreturn]] void rethrow_exception(exception_ptr __p) {
  if (!__p)
    throw bad_exception();

  auto* __rep = static_cast<__exception_ptr_storage*>(__p.__ptr_);
  auto __record_copy = __rep->__record_;
  auto& __cpp_record = reinterpret_cast<EHExceptionRecord&>(__record_copy);
  if (PER_IS_MSVC_PURE_OR_NATIVE_EH(&__cpp_record)) {
    const auto* __throw_info = __cpp_record.params.pThrowInfo;
    if (!__cpp_record.params.pExceptionObject || !__throw_info || !__throw_info->pCatchableTypeArray)
      terminate();

#if _EH_RELATIVE_TYPEINFO
    const auto __throw_image_base = reinterpret_cast<uintptr_t>(__cpp_record.params.pThrowImageBase);
    const auto* __catchable_type_array =
        reinterpret_cast<const CatchableTypeArray*>(__throw_image_base + __throw_info->pCatchableTypeArray);
#else // _EH_RELATIVE_TYPEINFO
    const auto* __catchable_type_array = __throw_info->pCatchableTypeArray;
#endif // _EH_RELATIVE_TYPEINFO

    if (__catchable_type_array->nCatchableTypes <= 0)
      terminate();

#if _EH_RELATIVE_TYPEINFO
    const auto* __type = reinterpret_cast<CatchableType*>(
        static_cast<uintptr_t>(__catchable_type_array->arrayOfCatchableTypes[0]) + __throw_image_base);
#else // _EH_RELATIVE_TYPEINFO
    const auto* __type = __throw_info->pCatchableTypeArray->arrayOfCatchableTypes[0];
#endif // _EH_RELATIVE_TYPEINFO

#pragma warning(suppress : 6255)
    void* __exception_buffer = alloca(__type->sizeOrOffset);
    __copy_exception_object(__exception_buffer, __cpp_record.params.pExceptionObject, __type
#if _EH_RELATIVE_TYPEINFO
                            ,
                            __throw_image_base
#endif // _EH_RELATIVE_TYPEINFO
    );

    __cpp_record.params.pExceptionObject = __exception_buffer;
  }

  // Under MSVC C++ exception handling (/EHsc), C Win32 API functions like RaiseException
  // are assumed not to throw synchronous C++ exceptions. As a result, stack unwinding
  // initiated by RaiseException does not execute local destructors in this frame.
  // We must explicitly reset __p here so that the reference count on the stored
  // exception record is properly decremented before unwinding begins.
  __p = nullptr;
  RaiseException(__record_copy.ExceptionCode, __record_copy.ExceptionFlags, __record_copy.NumberParameters,
                 __record_copy.ExceptionInformation);
  terminate();
}

nested_exception::nested_exception() noexcept : __ptr_(current_exception()) {}

nested_exception::~nested_exception() noexcept {}

[[noreturn]] void nested_exception::rethrow_nested() const {
  if (__ptr_ == nullptr)
    terminate();
  rethrow_exception(__ptr_);
}

} // namespace std
