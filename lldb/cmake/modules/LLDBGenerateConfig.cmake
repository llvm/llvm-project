# This file contains all the logic for running configure-time checks

include(CheckSymbolExists)
include(CheckIncludeFile)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CheckTypeSize)

set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
check_symbol_exists(ppoll poll.h HAVE_PPOLL)
set(CMAKE_REQUIRED_DEFINITIONS)
check_symbol_exists(sigaction signal.h HAVE_SIGACTION)
check_cxx_symbol_exists(accept4 "sys/socket.h" HAVE_ACCEPT4)

check_include_file(termios.h HAVE_TERMIOS_H)
check_include_files("sys/types.h;sys/event.h" HAVE_SYS_EVENT_H)

check_cxx_symbol_exists(process_vm_readv "sys/uio.h" HAVE_PROCESS_VM_READV)
check_cxx_symbol_exists(__NR_process_vm_readv "sys/syscall.h" HAVE_NR_PROCESS_VM_READV)

check_library_exists(compression compression_encode_buffer "" HAVE_LIBCOMPRESSION)

# These checks exist in LLVM's configuration, so I want to match the LLVM names
# so that the check isn't duplicated, but we translate them into the LLDB names
# so that I don't have to change all the uses at the moment.
set(LLDB_CONFIG_TERMIOS_SUPPORTED ${HAVE_TERMIOS_H})
if(NOT UNIX)
  set(LLDB_DISABLE_POSIX 1)
endif()

if (NOT LLDB_DISABLE_LIBEDIT)
  # Check if we libedit capable of handling wide characters (built with
  # '--enable-widec').
  set(CMAKE_REQUIRED_LIBRARIES ${libedit_LIBRARIES})
  set(CMAKE_REQUIRED_INCLUDES ${libedit_INCLUDE_DIRS})
  check_symbol_exists(el_winsertstr histedit.h LLDB_EDITLINE_USE_WCHAR)
  set(CMAKE_EXTRA_INCLUDE_FILES histedit.h)
  check_type_size(el_rfunc_t LLDB_EL_RFUNC_T_SIZE)
  if (LLDB_EL_RFUNC_T_SIZE STREQUAL "")
    set(LLDB_HAVE_EL_RFUNC_T 0)
  else()
    set(LLDB_HAVE_EL_RFUNC_T 1)
  endif()
  set(CMAKE_REQUIRED_LIBRARIES)
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_EXTRA_INCLUDE_FILES)
endif()

if(NOT LLDB_CONFIG_HEADER_INPUT)
 set(LLDB_CONFIG_HEADER_INPUT ${LLDB_INCLUDE_ROOT}/lldb/Host/Config.h.cmake)
endif()

if(NOT LLDB_CONFIG_HEADER_OUTPUT)
 set(LLDB_CONFIG_HEADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/lldb/Host/Config.h)
endif()

# This should be done at the end
configure_file(
  ${LLDB_CONFIG_HEADER_INPUT}
  ${LLDB_CONFIG_HEADER_OUTPUT}
  )
