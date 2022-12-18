// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_ABI_MICROSOFT
#error this header can only be used when targeting the MSVC ABI
#endif

// default_memory_resource()

extern "C" {

_LIBCPP_CRT_FUNC std::pmr::memory_resource* __cdecl _Aligned_set_default_resource(std::pmr::memory_resource*) noexcept;
_LIBCPP_CRT_FUNC std::pmr::memory_resource* __cdecl _Unaligned_set_default_resource(std::pmr::memory_resource*) noexcept;
_LIBCPP_CRT_FUNC std::pmr::memory_resource* __cdecl _Aligned_get_default_resource() noexcept;
_LIBCPP_CRT_FUNC std::pmr::memory_resource* __cdecl _Unaligned_get_default_resource() noexcept;
_LIBCPP_CRT_FUNC std::pmr::memory_resource* __cdecl _Aligned_new_delete_resource() noexcept;
_LIBCPP_CRT_FUNC std::pmr::memory_resource* __cdecl _Unaligned_new_delete_resource() noexcept;
_LIBCPP_CRT_FUNC std::pmr::memory_resource* __cdecl null_memory_resource() noexcept;

};

namespace std {

namespace pmr {

memory_resource* get_default_resource() noexcept {
    printf("Debug: compat get_default_resouce\n");
#ifdef __cpp_aligned_new
    return ::_Aligned_get_default_resource();
#else
    return ::_Unaligned_get_default_resource();
#endif
}

memory_resource* set_default_resource(memory_resource* __new_res) noexcept {
    printf("Debug: compat set_default_resouce\n");
#ifdef __cpp_aligned_new
    return ::_Aligned_set_default_resource(__new_res);
#else
    return ::_Unaligned_set_default_resource(__new_res);
#endif
}

memory_resource* new_delete_resource() noexcept {
#ifdef __cpp_aligned_new
    return ::_Aligned_new_delete_resource();
#else
    return ::_Unaligned_new_delete_resource();
#endif
}

memory_resource* null_memory_resource() noexcept {
    return ::null_memory_resource();
}

} // pmr

} // std
