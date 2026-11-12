//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>

#include <cstdlib>
#include <print>

#include <__system_error/system_error.h>

#include "filesystem/error.h"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <io.h>
#  include <versionhelpers.h>
#  include <windows.h>
#  include <winioctl.h>
#  include <winternl.h>
#elif __has_include(<unistd.h>)
#  include <unistd.h>
#  if defined(_NEWLIB_VERSION)
#    if defined(_POSIX_C_SOURCE) && __has_include(<stdio.h>)
#      include <stdio.h>
#      define HAS_FILENO_AND_ISATTY
#    endif
#  else
#    define HAS_FILENO_AND_ISATTY
#  endif
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

#ifdef _WIN32

using NtQueryVolumeInformationFileType = NTSTATUS
    NTAPI(HANDLE handle, PIO_STATUS_BLOCK ioStatus, PVOID information, ULONG length, ULONG informationClass);

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ns-wdm-_file_fs_device_information
struct FILE_FS_DEVICE_INFORMATION {
  DEVICE_TYPE DeviceType;
  ULONG Characteristics;
};

// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ne-wdm-_fsinfoclass
#  define FileFsDeviceInformation 4

static NtQueryVolumeInformationFileType* __get_NtQVIF_win8_or_greater() {
  // Windows 7 does not support querying the device type of a console handle, so returning nullptr indicates that this
  // approach is not supported.
  if (!IsWindows8OrGreater())
    return nullptr;
  return (NtQueryVolumeInformationFileType*)GetProcAddress(
      GetModuleHandleW(L"ntdll.dll"), "NtQueryVolumeInformationFile");
}

// https://learn.microsoft.com/en-us/windows/console/console-handles
// Exported by ntdll and can most efficiently query whether a HANDLE references a console.
static bool __is_handle_terminal_win8_or_greater(NtQueryVolumeInformationFileType* __fp, intptr_t __handle) {
  IO_STATUS_BLOCK __iosb;
  FILE_FS_DEVICE_INFORMATION __info;
  const auto __status =
      __fp(reinterpret_cast<void*>(__handle), &__iosb, &__info, sizeof(__info), FileFsDeviceInformation);
  return NT_SUCCESS(__status) && (__info.DeviceType == FILE_DEVICE_CONSOLE);
}

static bool __is_handle_terminal(intptr_t __handle) {
  // Note the Standard does this in one call, but it's unclear whether
  // an invalid handle is allowed when calling GetConsoleMode.
  // https://learn.microsoft.com/en-us/windows/console/getconsolemode
  unsigned long __mode;
  return GetConsoleMode(reinterpret_cast<void*>(__handle), &__mode);
}

_LIBCPP_EXPORTED_FROM_ABI bool __is_windows_terminal(FILE* __stream) {
  // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/get-osfhandle?view=msvc-170
  intptr_t __handle = _get_osfhandle(fileno(__stream));
  if (__handle == -1)
    return false;
  static auto __fp = __get_NtQVIF_win8_or_greater();
  return __fp ? __is_handle_terminal_win8_or_greater(__fp, __handle) : __is_handle_terminal(__handle);
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
_LIBCPP_EXPORTED_FROM_ABI void
__write_to_windows_console([[maybe_unused]] FILE* __stream, [[maybe_unused]] wstring_view __view) {
  // https://learn.microsoft.com/en-us/windows/console/writeconsole
  if (WriteConsoleW(reinterpret_cast<void*>(_get_osfhandle(fileno(__stream))), // clang-format aid
                    __view.data(),
                    __view.size(),
                    nullptr,
                    nullptr) == 0) {
    std::__throw_system_error(filesystem::detail::get_last_error(), "failed to write formatted output");
  }
}
#  endif // _LIBCPP_HAS_WIDE_CHARACTERS

#elif defined(HAS_FILENO_AND_ISATTY) && _LIBCPP_AVAILABILITY_MINIMUM_HEADER_VERSION < 23

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wmissing-prototypes")
_LIBCPP_EXPORTED_FROM_ABI bool __is_posix_terminal(FILE* __stream) { return isatty(fileno(__stream)); }
_LIBCPP_DIAGNOSTIC_POP
#endif // _WIN32

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
