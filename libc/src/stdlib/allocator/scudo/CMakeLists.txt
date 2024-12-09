set(SCUDO_DEPS "")

include(${LIBC_SOURCE_DIR}/../compiler-rt/cmake/Modules/AllSupportedArchDefs.cmake)

# scudo distinguishes riscv32 and riscv64, so we need to translate the architecture
# set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO ${LIBC_TARGET_ARCHITECTURE})
if (LIBC_TARGET_OS_IS_DARWIN AND (LIBC_TARGET_ARCHITECTURE STREQUAL "arm"))
  set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO arm64)
else()
  set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO ${LIBC_TARGET_ARCHITECTURE})
endif()
  if(LIBC_TARGET_ARCHITECTURE_IS_RISCV64)
    set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO riscv64)
  elseif(LIBC_TARGET_ARCHITECTURE_IS_RISCV32)
    set(LIBC_TARGET_ARCHITECTURE_FOR_SCUDO riscv32)
endif()

if(NOT (LIBC_TARGET_ARCHITECTURE_FOR_SCUDO IN_LIST ALL_SCUDO_STANDALONE_SUPPORTED_ARCH))
  message(FATAL_ERROR "Architecture ${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO} is not supported by SCUDO.
    Either disable LLVM_LIBC_INCLUDE_SCUDO or change your target architecture.")
endif()

list(APPEND SCUDO_DEPS RTScudoStandalone.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
  RTScudoStandaloneCWrappers.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO})
    
if (COMPILER_RT_BUILD_GWP_ASAN)
  list(APPEND SCUDO_DEPS
    RTGwpAsan.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
    RTGwpAsanBacktraceLibc.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
    RTGwpAsanSegvHandler.${LIBC_TARGET_ARCHITECTURE_FOR_SCUDO}
  )
endif()

add_entrypoint_external(
  @libc.src.stdlib.malloc
  DEPENDS
    ${SCUDO_DEPS}
)
    
add_entrypoint_external(
  @libc.src.stdlib.calloc
  DEPENDS
    ${SCUDO_DEPS}
)

add_entrypoint_external(
  @libc.src.stdlib.realloc
  DEPENDS
    ${SCUDO_DEPS}
)

add_entrypoint_external(
  @libc.src.stdlib.aligned_alloc
  DEPENDS
    ${SCUDO_DEPS}
)
add_entrypoint_external(
  @libc.src.stdlib.free
  DEPENDS
    ${SCUDO_DEPS}
)
add_entrypoint_external(
  @libc.src.stdlib.mallopt
  DEPENDS
    ${SCUDO_DEPS}
)
