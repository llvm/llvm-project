add_subdirectory(generic)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_ARCHITECTURE})
  add_subdirectory(${LIBC_TARGET_ARCHITECTURE})
endif()

# ------------------------------------------------------------------------------
# Common Utilities
# ------------------------------------------------------------------------------

add_header_library(
  utils
  HDRS
    utils.h
  DEPENDS
    libc.src.__support.common
    libc.src.__support.CPP.bit
    libc.src.__support.CPP.cstddef
    libc.src.__support.CPP.type_traits
    libc.src.__support.macros.config
    libc.src.__support.macros.optimization
    libc.src.__support.macros.properties.architectures
    libc.src.__support.macros.properties.compiler
)

add_header_library(op_builtin HDRS op_builtin.h DEPENDS .utils)
add_header_library(op_generic HDRS op_generic.h DEPENDS .utils)
add_header_library(op_x86     HDRS op_x86.h     DEPENDS .utils)
add_header_library(op_aarch64 HDRS op_aarch64.h DEPENDS .utils)
add_header_library(op_riscv   HDRS op_riscv.h   DEPENDS .utils)

# ------------------------------------------------------------------------------
# Dispatch Logic
# ------------------------------------------------------------------------------

function(add_memory_utils_shim name)
  set(depends_list "")
  get_fq_target_name("${LIBC_TARGET_ARCHITECTURE}.${name}" fq_machine_specific_target_name)
  if(TARGET ${fq_machine_specific_target_name})
    list(APPEND depends_list ${fq_machine_specific_target_name})
  else()
    get_fq_target_name("generic.${name}" fq_generic_target_name)
    if(TARGET ${fq_generic_target_name})
      list(APPEND depends_list ${fq_generic_target_name})
    endif()
  endif()

  add_header_library(
    ${name}
    HDRS ${name}.h
    DEPENDS ${depends_list}
  )
endfunction()

add_memory_utils_shim(inline_memcpy)
add_memory_utils_shim(inline_memmove)
add_memory_utils_shim(inline_memcmp)
add_memory_utils_shim(inline_memset)
add_memory_utils_shim(inline_bcmp)

# ------------------------------------------------------------------------------
# No-dispatch implementations
# ------------------------------------------------------------------------------

add_header_library(
  inline_bzero
  HDRS inline_bzero.h
  DEPENDS .inline_memset
)

add_header_library(
  inline_strcmp
  HDRS inline_strcmp.h
)

add_header_library(
  inline_strstr
  HDRS inline_strstr.h
)

add_header_library(
  inline_memmem
  HDRS inline_memmem.h
)
