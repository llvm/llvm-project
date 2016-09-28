//===-------------------------- __cxxabi_config.h -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef ____CXXABI_CONFIG_H
#define ____CXXABI_CONFIG_H

#if defined(__arm__) && !defined(__USING_SJLJ_EXCEPTIONS__) &&                 \
    !defined(__ARM_DWARF_EH__)
#define LIBCXXABI_ARM_EHABI 1
#else
#define LIBCXXABI_ARM_EHABI 0
#endif

#if !defined(__has_attribute)
#define __has_attribute(_attribute_) 0
#endif

#if defined(_WIN32)
 #if defined(_LIBCXXABI_DISABLE_DLL_IMPORT_EXPORT)
  #define _LIBCXXABI_HIDDEN
  #define _LIBCXXABI_DATA_VIS
  #define _LIBCXXABI_FUNC_VIS
  #define _LIBCXXABI_TYPE_VIS
 #elif defined(_LIBCXXABI_BUILDING_LIBRARY)
  #define _LIBCXXABI_HIDDEN
  #define _LIBCXXABI_DATA_VIS __declspec(dllexport)
  #define _LIBCXXABI_FUNC_VIS __declspec(dllexport)
  #define _LIBCXXABI_TYPE_VIS __declspec(dllexport)
 #else
  #define _LIBCXXABI_HIDDEN
  #define _LIBCXXABI_DATA_VIS __declspec(dllimport)
  #define _LIBCXXABI_FUNC_VIS __declspec(dllimport)
  #define _LIBCXXABI_TYPE_VIS __declspec(dllimport)
 #endif
#else
 #define _LIBCXXABI_HIDDEN __attribute__((__visibility__("hidden")))
 #define _LIBCXXABI_DATA_VIS __attribute__((__visibility__("default")))
 #define _LIBCXXABI_FUNC_VIS __attribute__((__visibility__("default")))
 #if __has_attribute(__type_visibility__)
  #define _LIBCXXABI_TYPE_VIS __attribute__((__type_visibility__("default")))
 #else
  #define _LIBCXXABI_TYPE_VIS __attribute__((__visibility__("default")))
 #endif
#endif

#endif // ____CXXABI_CONFIG_H
