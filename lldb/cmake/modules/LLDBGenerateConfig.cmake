# This file contains all the logic for running configure-time checks

include(CheckSymbolExists)
include(CheckIncludeFile)
include(CheckIncludeFiles)

set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
check_symbol_exists(ppoll poll.h HAVE_PPOLL)
set(CMAKE_REQUIRED_DEFINITIONS)
check_symbol_exists(sigaction signal.h HAVE_SIGACTION)

check_include_file(termios.h HAVE_TERMIOS_H)
check_include_files("sys/types.h;sys/event.h" HAVE_SYS_EVENT_H)

# These checks exist in LLVM's configuration, so I want to match the LLVM names
# so that the check isn't duplicated, but we translate them into the LLDB names
# so that I don't have to change all the uses at the moment.
set(LLDB_CONFIG_TERMIOS_SUPPORTED ${HAVE_TERMIOS_H})
if(NOT UNIX)
  set(LLDB_DISABLE_POSIX 1)
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
