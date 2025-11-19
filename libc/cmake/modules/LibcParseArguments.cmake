set(LLVM_LIBC_OPTIONAL_ARGS
  ALIAS
  PUBLIC
  NO_GPU_BUNDLE
  NO_RUN_POSTBUILD
  C_TEST
  NEED_MPFR
  NEED_MPC
  IS_GPU_BENCHMARK
)

set(LLVM_LIBC_SINGLE_VALUE_ARGS
  CXX_STANDARD
  SUITE
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
  ARGS
  LOADER_ARGS
)

foreach(arg_list LLVM_LIBC_OPTIONAL_ARGS LLVM_LIBC_SINGLE_VALUE_ARGS LLVM_LIBC_MULTI_VALUE_ARGS)
  list(TRANSFORM ${arg_list}
    PREPEND "OVERLAY_"
    OUTPUT_VARIABLE ${arg_list}_OVERLAY
  )
  list(TRANSFORM ${arg_list}
    PREPEND "FULL_BUILD_"
    OUTPUT_VARIABLE ${arg_list}_FULL_BUILD
  )
  set(${arg_list}_COMPLETE ${${arg_list}} ${${arg_list}_OVERLAY} ${${arg_list}_FULL_BUILD})
endforeach()

macro(llvm_libc_parse_arguments name_prefix)
  cmake_parse_arguments(
    ${name_prefix}
    "${LLVM_LIBC_OPTIONAL_ARGS_COMPLETE}"
    "${LLVM_LIBC_SINGLE_VALUE_ARGS_COMPLETE}"
    "${LLVM_LIBC_MULTI_VALUE_ARGS_COMPLETE}"
    ${ARGN}
  )

  # Collect overlay and full build args
  foreach(argument IN LISTS LLVM_LIBC_OPTIONAL_ARGS LLVM_LIBC_SINGLE_VALUE_ARGS LLVM_LIBC_MULTI_VALUE_ARGS)
    if(LLVM_LIBC_FULL_BUILD)
      if(${name_prefix}_${argument}_FULL_BUILD)
        list(APPEND ${name_prefix}_${argument} ${${name_prefix}_${argument}_FULL_BUILD})
      endif()
    else()
      if(${name_prefix}_${argument}_OVERLAY)
        list(APPEND ${name_prefix}_${argument} ${${name_prefix}_${argument}_OVERLAY})
      endif()
    endif()
  endforeach()
endmacro()

# Forward all arguments that can be used for llvm_libc_parse_arguments again.
# Assume that *_OVERLAY and *_FULL_BUILD args have been merged properly.
macro(forward_arguments name_prefix output)
  set(${output} "")
  
  foreach(argument ${LLVM_LIBC_OPTIONAL_ARGS})
    if(${name_prefix}_${argument})
      list(APPEND output ${argument})
    endif()
  endforeach()

  foreach(argument ${LLVM_LIBC_SINGLE_VALUE_ARGS} ${LLVM_LIBC_MULTI_VALUE_ARGS})
    if(${name_prefix}_${argument})
      list(APPEND output ${argument} "${${name_prefix}_${argument}}")
    endif()
  endforeach()
endmacro()
