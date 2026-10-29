# This file provides a unified argument parsing facility for LLVM-libc CMake
# rules (such as add_libc_test, create_libc_test, add_entrypoint_object, etc.).
#
# Every argument declared in `LLVM_LIBC_OPTIONAL_ARGS`,
# `LLVM_LIBC_SINGLE_VALUE_ARGS`, or `LLVM_LIBC_MULTI_VALUE_ARGS` automatically
# receives two mode-specific variants:
#   OVERLAY_<arg>
#   FULL_BUILD_<arg>
#
# When `llvm_libc_parse_arguments` is called:
#   1. It parses all base, overlay, and full-build argument variants using
#      cmake_parse_arguments.
#   2. It selects the active build mode from `LLVM_LIBC_FULL_BUILD`.
#   3. It resolves mode-specific options into the canonical `<prefix>_<arg>`
#      variables:
#        - Optional and single-value arguments may be specified either via the
#          base `<arg>` or via `OVERLAY_<arg>` / `FULL_BUILD_<arg>` (specifying
#          both is an error).
#        - Multi-value arguments append `OVERLAY_<arg>` or `FULL_BUILD_<arg>` to
#          the base `<arg>` list.
#
# Examples:
#
# 1. Mode-specific dependencies or compile options:
#
#      add_libc_test(
#        sin_test
#        SUITE libc-math-tests
#        SRCS sin_test.cpp
#        DEPENDS libc.src.math.sin
#        OVERLAY_DEPENDS libc.test.UnitTest.OverlaySupport
#        FULL_BUILD_DEPENDS libc.test.UnitTest.HermeticSupport
#        FULL_BUILD_COMPILE_OPTIONS -ffreestanding
#      )
#
#    - In overlay mode (LLVM_LIBC_FULL_BUILD=OFF):
#        TEST_DEPENDS will contain:
#          libc.src.math.sin;libc.test.UnitTest.OverlaySupport
#
#    - In full-build mode (LLVM_LIBC_FULL_BUILD=ON):
#        TEST_DEPENDS will contain:
#          libc.src.math.sin;libc.test.UnitTest.HermeticSupport
#        TEST_COMPILE_OPTIONS will contain:
#          -ffreestanding
#
# 2. Mode-restricted targets:
#
#      add_libc_test(
#        loader_test
#        FULL_BUILD_ONLY
#        SUITE libc-loader-tests
#        SRCS loader_test.cpp
#        DEPENDS ...
#      )
#
#    - When parsed in overlay mode:
#        TEST_FULL_BUILD_ONLY evaluates to TRUE, allowing the caller to
#        skip defining the target. Similarly, `OVERLAY_ONLY` skips the
#        target in full-build mode.
#
# 3. Forwarding merged arguments:
#
#    `forward_arguments(name_prefix output_var)` re-emits only the canonical
#    (merged) arguments, dropping the mode-specific prefixes. This allows
#    outer wrapper functions to pass fully-resolved arguments down to inner
#    rules without repeating mode resolution:
#
#      function(add_libc_test test_name)
#        llvm_libc_parse_arguments(LIBC_TEST ${ARGN})
#        ...
#        forward_arguments(LIBC_TEST forwarded_args)
#        _add_libc_test(${test_name} ${forwarded_args} ${LIBC_TEST_UNPARSED_ARGUMENTS})
#      endfunction()

# Common options that control target inclusion by mode. These do NOT receive
# OVERLAY_ or FULL_BUILD_ prefixes.
set(LLVM_LIBC_COMMON_OPTIONAL_ARGS
  FULL_BUILD_ONLY
  OVERLAY_ONLY
)

# Target configuration options that can have mode-specific variants.
set(LLVM_LIBC_OPTIONAL_ARGS
  ALIAS
  PUBLIC
  PROXY
  REDIRECTED
  SKIPPED
  NO_GPU_BUNDLE
  C_TEST
  NEED_MPFR
  NEED_MPFR_F80
  NEED_MPFR_F128
  NEED_MPC
  FP_TEST
  INTEGRATION_TEST
  BUILD_ONLY
  ROOT
  PACKAGE_VARIANTS
  DISPATCH_VARIANTS
)

set(LLVM_LIBC_SINGLE_VALUE_ARGS
  CXX_STANDARD
  SUITE
  PARENT
  STARTUP
  CREATE_TARGET
  CREATE_TARGET_FUNCTION
  HDR
  DEST_HDR
  YAML_FILE
  GEN_HDR
  NAME
)

set(LLVM_LIBC_MULTI_VALUE_ARGS
  HDRS
  SRCS
  COMPILE_OPTIONS
  LINK_OPTIONS
  LINK_LIBRARIES
  ENV
  DEPENDS
  FLAGS
  ADD_FLAGS
  ARGS
  LOADER_ARGS
)

# Generate OVERLAY_<arg> and FULL_BUILD_<arg> variants for every base argument.
foreach(arg_list LLVM_LIBC_OPTIONAL_ARGS LLVM_LIBC_SINGLE_VALUE_ARGS LLVM_LIBC_MULTI_VALUE_ARGS)
  list(TRANSFORM ${arg_list}
    PREPEND "OVERLAY_"
    OUTPUT_VARIABLE OVERLAY_${arg_list}
  )
  list(TRANSFORM ${arg_list}
    PREPEND "FULL_BUILD_"
    OUTPUT_VARIABLE FULL_BUILD_${arg_list}
  )
  set(${arg_list}_COMPLETE ${${arg_list}} ${OVERLAY_${arg_list}} ${FULL_BUILD_${arg_list}})
endforeach()

# Parse arguments and automatically merge mode-specific options into the
# canonical <name_prefix>_<argument> variables in the caller's scope.
function(llvm_libc_parse_arguments name_prefix)
  cmake_parse_arguments(
    ARG
    "${LLVM_LIBC_COMMON_OPTIONAL_ARGS};${LLVM_LIBC_OPTIONAL_ARGS_COMPLETE}"
    "${LLVM_LIBC_SINGLE_VALUE_ARGS_COMPLETE}"
    "${LLVM_LIBC_MULTI_VALUE_ARGS_COMPLETE}"
    ${ARGN}
  )

  if(LLVM_LIBC_FULL_BUILD)
    set(mode_prefix "FULL_BUILD")
  else()
    set(mode_prefix "OVERLAY")
  endif()

  # Resolve mode-specific optional and single-value arguments.
  # Specifying both the base argument and a mode-specific variant is forbidden.
  foreach(argument IN LISTS LLVM_LIBC_OPTIONAL_ARGS LLVM_LIBC_SINGLE_VALUE_ARGS)
    if(ARG_${argument} AND (ARG_OVERLAY_${argument} OR ARG_FULL_BUILD_${argument}))
      message(FATAL_ERROR
        "Cannot specify both ${argument} and OVERLAY_${argument}/FULL_BUILD_${argument}."
      )
    endif()
    if(ARG_${mode_prefix}_${argument})
      set(ARG_${argument} "${ARG_${mode_prefix}_${argument}}")
    endif()
  endforeach()

  # Merge mode-specific multi-value arguments (append to base list).
  foreach(argument IN LISTS LLVM_LIBC_MULTI_VALUE_ARGS)
    if(ARG_${mode_prefix}_${argument})
      list(APPEND ARG_${argument} ${ARG_${mode_prefix}_${argument}})
    endif()
  endforeach()

  foreach(argument
    IN LISTS
    LLVM_LIBC_COMMON_OPTIONAL_ARGS
    LLVM_LIBC_OPTIONAL_ARGS
    LLVM_LIBC_SINGLE_VALUE_ARGS
    LLVM_LIBC_MULTI_VALUE_ARGS
    ITEMS
    UNPARSED_ARGUMENTS
    KEYWORDS_MISSING_VALUES
  )
    set(${name_prefix}_${argument} "${ARG_${argument}}" PARENT_SCOPE)
  endforeach()
endfunction()

# Forward all arguments that can be used for llvm_libc_parse_arguments again.
# Re-emits only the canonical arguments, assuming that OVERLAY_* and FULL_BUILD_*
# args have already been merged properly.
function(forward_arguments name_prefix output)
  set(forwarded "")

  foreach(argument IN LISTS LLVM_LIBC_COMMON_OPTIONAL_ARGS LLVM_LIBC_OPTIONAL_ARGS)
    if(${name_prefix}_${argument})
      list(APPEND forwarded ${argument})
    endif()
  endforeach()

  foreach(argument IN LISTS LLVM_LIBC_SINGLE_VALUE_ARGS LLVM_LIBC_MULTI_VALUE_ARGS)
    if(${name_prefix}_${argument})
      list(APPEND forwarded ${argument} "${${name_prefix}_${argument}}")
    endif()
  endforeach()

  set(${output} "${forwarded}" PARENT_SCOPE)
endfunction()
