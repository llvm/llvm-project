set(SCUDO_DEPS "")

if(LLVM_LIBC_INCLUDE_SCUDO)
  include(../../compiler-rt/cmake/Modules/AllSupportedArchDefs.cmake)
  if(NOT (LIBC_TARGET_ARCHITECTURE IN_LIST ALL_SCUDO_STANDALONE_SUPPORTED_ARCH))
    message(FATAL_ERROR "Architecture ${LIBC_TARGET_ARCHITECTURE} is not supported by SCUDO. 
Either disable LLVM_LIBC_INCLUDE_SCUDO or change your target architecture.")
  endif()
  list(APPEND SCUDO_DEPS RTScudoStandalone.${LIBC_TARGET_ARCHITECTURE} RTScudoStandaloneCWrappers.${LIBC_TARGET_ARCHITECTURE})
  if((LIBC_TARGET_ARCHITECTURE IN_LIST ALL_GWP_ASAN_SUPPORTED_ARCH) AND COMPILER_RT_BUILD_GWP_ASAN)
    list(APPEND SCUDO_DEPS RTGwpAsan.${LIBC_TARGET_ARCHITECTURE} 
                            RTGwpAsanBacktraceLibc.${LIBC_TARGET_ARCHITECTURE} 
                            RTGwpAsanSegvHandler.${LIBC_TARGET_ARCHITECTURE})
  elseif(COMPILER_RT_BUILD_GWP_ASAN)
    message(WARNING "Architecture ${LIBC_TARGET_ARCHITECTURE} is not supported by GWP-ASan. Skipping.")
  endif()
endif()

add_entrypoint_library(
  llvmlibc
  DEPENDS
  ${TARGET_LLVMLIBC_ENTRYPOINTS}
  EXT_DEPS
  ${SCUDO_DEPS}
)

if(LLVM_ENABLE_PER_TARGET_RUNTIME_DIR)
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX}/${LLVM_DEFAULT_TARGET_TRIPLE})
else()
  set(LIBC_INSTALL_LIBRARY_DIR lib${LLVM_LIBDIR_SUFFIX})
endif()

install(
  TARGETS llvmlibc
  ARCHIVE DESTINATION "${LIBC_INSTALL_LIBRARY_DIR}"
  COMPONENT llvmlibc
)

add_custom_target(
    install-llvmlibc
    DEPENDS llvmlibc
    COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=llvmlibc
            -P "${LIBC_BINARY_DIR}/cmake_install.cmake"
)
