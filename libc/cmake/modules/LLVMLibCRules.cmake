# Unified CMake diagnostic logging configuration.
# Levels for LIBC_CMAKE_VERBOSE_LOGGING:
#   OFF:          Quiet configure (only summary, warnings, and errors).
#   ON / BASIC:   Skipped entrypoints/tests/suites, config overrides, alias fallbacks.
#   TARGETS:      Includes BASIC, plus target creation across all rules, alias chains, and flag variants.
#   DEPS:         Includes TARGETS, plus full dependency DAG, flag projection, and argument merges.
set(LIBC_CMAKE_VERBOSE_LOGGING "OFF" CACHE STRING
  "Verbose CMake logging level: OFF, ON (or BASIC), TARGETS, DEPS."
)
set(LIBC_CMAKE_VERBOSE_TARGET_FILTER "" CACHE STRING
  "Optional regex filter restricting libc_log output to matching messages/targets."
)

if(NOT LIBC_CMAKE_VERBOSE_LOGGING)
  set(_LIBC_LOG_LEVEL 0)
elseif(LIBC_CMAKE_VERBOSE_LOGGING STREQUAL "DEPS")
  set(_LIBC_LOG_LEVEL 3)
elseif(LIBC_CMAKE_VERBOSE_LOGGING STREQUAL "TARGETS")
  set(_LIBC_LOG_LEVEL 2)
else()
  set(_LIBC_LOG_LEVEL 1)
endif()

# Unified CMake diagnostic logger.
# Usage:
#   libc_log(<LEVEL> <message>...)
function(libc_log level)
  if(NOT _LIBC_LOG_LEVEL)
    return()
  endif()

  if(level STREQUAL "DEPS")
    set(req_level 3)
  elseif(level STREQUAL "TARGETS")
    set(req_level 2)
  else()
    set(req_level 1)
  endif()

  if(_LIBC_LOG_LEVEL LESS req_level)
    return()
  endif()

  string(JOIN " " msg_text ${ARGN})
  if(LIBC_CMAKE_VERBOSE_TARGET_FILTER)
    if(NOT "${msg_text}" MATCHES "${LIBC_CMAKE_VERBOSE_TARGET_FILTER}")
      return()
    endif()
  endif()

  message(STATUS "${msg_text}")
endfunction()

include(LibcParseArguments)
include(LLVMLibCCompileOptionRules)
include(LLVMLibCTargetNameUtils)
include(LLVMLibCFlagRules)
include(LLVMLibCObjectRules)
include(LLVMLibCLibraryRules)
include(LLVMLibCTestRules)
