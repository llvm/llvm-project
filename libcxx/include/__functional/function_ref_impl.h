//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <__config>
#include <__functional/invoke.h>
#include <__memory/addressof.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_const.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_member_pointer.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_void.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/remove_pointer.h>
#include <__type_traits/remove_reference.h>
#include <__utility/forward.h>
#include <__utility/nontype.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26

template <class...>
class function_ref;

template <class _Rp, class... _ArgTypes>
class function_ref<_Rp(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT)> {
private:
#  if _LIBCPP_FUNCTION_REF_NOEXCEPT == true
  template <class... _Tp>
  static constexpr bool __is_invocable_using = is_nothrow_invocable_r_v<_Rp, _Tp..., _ArgTypes...>;
#  else
  template <class... _Tp>
  static constexpr bool __is_invocable_using = is_invocable_r_v<_Rp, _Tp..., _ArgTypes...>;
#  endif

  // use a union instead of a plain `void*` to avoid dropping const qualifiers and casting function pointers to data
  // pointers
  union __storage_t {
    void* __obj_ptr;
    void const* __obj_const_ptr;
    void (*__fn_ptr)();

    _LIBCPP_HIDE_FROM_ABI constexpr explicit __storage_t() noexcept : __obj_ptr(nullptr) {}

    template <class _Tp>
    _LIBCPP_HIDE_FROM_ABI constexpr explicit __storage_t(_Tp* __ptr) noexcept {
      if constexpr (is_object_v<_Tp>) {
        if constexpr (is_const_v<_Tp>) {
          __obj_const_ptr = __ptr;
        } else {
          __obj_ptr = __ptr;
        }
      } else {
        static_assert(is_function_v<_Tp>);
        __fn_ptr = reinterpret_cast<void (*)()>(__ptr);
      }
    }
  } __storage_;

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI static constexpr auto __get(__storage_t __storage) {
    if constexpr (is_object_v<_Tp>) {
      if constexpr (is_const_v<_Tp>) {
        return static_cast<_Tp*>(__storage.__obj_const_ptr);
      } else {
        return static_cast<_Tp*>(__storage.__obj_ptr);
      }
    } else {
      static_assert(is_function_v<_Tp>);
      return reinterpret_cast<_Tp*>(__storage.__fn_ptr);
    }
  }

  using __call_t = _Rp (*)(__storage_t, _ArgTypes&&...) noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT);
  __call_t __call_;

public:
  template <class _Fp>
    requires is_function_v<_Fp> && __is_invocable_using<_Fp>
  _LIBCPP_HIDE_FROM_ABI function_ref(_Fp* __fn_ptr) noexcept
      : __storage_(__fn_ptr),
        __call_([](__storage_t __storage, _ArgTypes&&... __args) static noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT) -> _Rp {
          return __get<_Fp>(__storage)(std::forward<_ArgTypes>(__args)...);
        }) {
    _LIBCPP_ASSERT_UNCATEGORIZED(__fn_ptr != nullptr, "the function pointer should not be a nullptr");
  }

  template <class _Fp, class _Tp = remove_reference_t<_Fp>>
    requires(!__is_function_ref<remove_cvref_t<_Fp>> && !is_member_pointer_v<_Tp> &&
             __is_invocable_using<_LIBCPP_FUNCTION_REF_CV _Tp&>)
  _LIBCPP_HIDE_FROM_ABI function_ref(_Fp&& __obj) noexcept
      : __storage_(std::addressof(__obj)),
        __call_([](__storage_t __storage, _ArgTypes&&... __args) static noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT) -> _Rp {
          _LIBCPP_FUNCTION_REF_CV _Tp& __obj = *__get<_Tp>(__storage);
          return __obj(std::forward<_ArgTypes>(__args)...);
        }) {}

  template <auto _Fn>
    requires __is_invocable_using<decltype(_Fn)>
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(nontype_t<_Fn>) noexcept
      : __call_([](__storage_t, _ArgTypes&&... __args) static noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT) -> _Rp {
          return std::invoke_r<_Rp>(_Fn, std::forward<_ArgTypes>(__args)...);
        }) {
    if constexpr (is_pointer_v<decltype(_Fn)> || is_member_pointer_v<decltype(_Fn)>) {
      static_assert(_Fn != nullptr, "the function pointer should not be a nullptr");
    }
  }

  template <auto _Fn, class _Up, class _Tp = remove_reference_t<_Up>>
    requires(!is_rvalue_reference_v<_Up &&>) && __is_invocable_using<decltype(_Fn), _LIBCPP_FUNCTION_REF_CV _Tp&>
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(nontype_t<_Fn>, _Up&& __obj) noexcept
      : __storage_(std::addressof(__obj)),
        __call_([](__storage_t __storage, _ArgTypes&&... __args) static noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT) -> _Rp {
          _LIBCPP_FUNCTION_REF_CV _Tp& __obj = *__get<_Tp>(__storage);
          return std::invoke_r<_Rp>(_Fn, __obj, std::forward<_ArgTypes>(__args)...);
        }) {
    if constexpr (is_pointer_v<decltype(_Fn)> || is_member_pointer_v<decltype(_Fn)>) {
      static_assert(_Fn != nullptr, "the function pointer should not be a nullptr");
    }
  }

  template <auto _Fn, class _Tp>
    requires __is_invocable_using<decltype(_Fn), _LIBCPP_FUNCTION_REF_CV _Tp*>
  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(nontype_t<_Fn>, _LIBCPP_FUNCTION_REF_CV _Tp* __obj_ptr) noexcept
      : __storage_(__obj_ptr),
        __call_([](__storage_t __storage, _ArgTypes&&... __args) static noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT) -> _Rp {
          auto __obj = __get<_LIBCPP_FUNCTION_REF_CV _Tp>(__storage);
          return std::invoke_r<_Rp>(_Fn, __obj, std::forward<_ArgTypes>(__args)...);
        }) {
    if constexpr (is_pointer_v<decltype(_Fn)> || is_member_pointer_v<decltype(_Fn)>) {
      static_assert(_Fn != nullptr, "the function pointer should not be a nullptr");
    }

    if constexpr (is_member_pointer_v<decltype(_Fn)>) {
      _LIBCPP_ASSERT_UNCATEGORIZED(__obj_ptr != nullptr, "the object pointer should not be a nullptr");
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr function_ref(const function_ref&) noexcept = default;

  _LIBCPP_HIDE_FROM_ABI constexpr function_ref& operator=(const function_ref&) noexcept = default;

  template <class _Tp>
    requires(!__is_function_ref<_Tp>) && (!is_pointer_v<_Tp>) && (!__is_nontype_t<_Tp>)
  _LIBCPP_HIDE_FROM_ABI function_ref& operator=(_Tp) = delete;

  _LIBCPP_HIDE_FROM_ABI constexpr _Rp operator()(_ArgTypes... __args) const noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT) {
    return __call_(__storage_, std::forward<_ArgTypes>(__args)...);
  }
};

template <class _Tp, class _Rp, class _Gp, class... _ArgTypes>
struct __function_ref_bind<_Rp (_Gp::*)(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT),
                           _Tp> {
  using type = _Rp(_ArgTypes...) noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT);
};

template <class _Tp, class _Rp, class _Gp, class... _ArgTypes>
struct __function_ref_bind<_Rp (_Gp::*)(_ArgTypes...) _LIBCPP_FUNCTION_REF_CV & noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT),
                           _Tp> {
  using type = _Rp(_ArgTypes...) noexcept(_LIBCPP_FUNCTION_REF_NOEXCEPT);
};

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_END_NAMESPACE_STD
