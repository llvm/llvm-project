# Unified CMake diagnostic logging configuration.
# Levels for LIBC_CMAKE_VERBOSE_LOGGING:
#   OFF:          Quiet configure (only summary, warnings, and errors).
#   ON / BASIC:   Skipped entrypoints/tests/suites, config overrides, alias fallbacks.
#   TARGETS:      Includes BASIC, plus target creation across all rules, alias chains, and flag variants.
#   DEPS:         Includes TARGETS, plus full dependency DAG, flag projection, and argument overrides.
set(LIBC_CMAKE_VERBOSE_LOGGING "OFF" CACHE STRING
  "Verbose CMake logging level: OFF, ON (or BASIC), TARGETS, DEPS."
)
set(LIBC_CMAKE_VERBOSE_TARGET_FILTER "" CACHE STRING
  "Optional regex filter restricting TARGETS and DEPS logging to matching target names."
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

function(_libc_log_impl level)
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

  set(msg_args ${ARGN})
  set(filter_subject "")
  list(LENGTH msg_args num_args)
  if(num_args GREATER_EQUAL 2)
    list(GET msg_args 0 first_arg)
    if(first_arg STREQUAL "TARGET")
      list(GET msg_args 1 filter_subject)
      list(REMOVE_AT msg_args 0 1)
    endif()
  endif()

  string(JOIN "" msg_text ${msg_args})
  if(LIBC_CMAKE_VERBOSE_TARGET_FILTER)
    if(NOT filter_subject)
      set(filter_subject "${msg_text}")
    endif()
    if(NOT "${filter_subject}" MATCHES "${LIBC_CMAKE_VERBOSE_TARGET_FILTER}")
      return()
    endif()
  endif()

  message(STATUS "${msg_text}")
endfunction()

# Unified CMake diagnostic logger.
# Usage:
#   libc_log(<LEVEL> [TARGET <target_name>] <message>...)
macro(libc_log _libc_log_level)
  if(_LIBC_LOG_LEVEL)
    _libc_log_impl("${_libc_log_level}" ${ARGN})
  endif()
endmacro()

include(LibcParseArguments)
include(LLVMLibCCompileOptionRules)
include(LLVMLibCTargetNameUtils)
include(LLVMLibCFlagRules)
include(LLVMLibCObjectRules)
include(LLVMLibCLibraryRules)
include(LLVMLibCTestRules)
