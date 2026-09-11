set(LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})

if(LIBCLC_TARGET_ARCH IN_LIST LIBCLC_ARCHS_AMDGPU)
  set(LIBCLC_TARGET_CPU "gfx900")
else()
  set(LIBCLC_TARGET_CPU "")
endif()

include(CheckCXXCompilerFlag)

set(LIBCLC_TEST_ARCHITECTURE "" CACHE STRING
  "Architecture for libclc execution tests; 'native' or a specific name")

set(LIBCLC_OFFLOAD_LIBDIR "")

if(NOT LIBCLC_TEST_ARCHITECTURE)
  check_cxx_compiler_flag("-march=native" LIBCLC_HAS_NATIVE_DEVICE)
  if(LIBCLC_HAS_NATIVE_DEVICE)
    set(LIBCLC_TEST_ARCHITECTURE "native")
  endif()
endif()

if(LIBCLC_TEST_ARCHITECTURE)
  find_library(LIBCLC_OFFLOAD_LIB LLVMOffload
    PATHS "${LLVM_LIBRARY_DIR}"
    PATH_SUFFIXES "${LLVM_HOST_TRIPLE}"
    NO_DEFAULT_PATH)
  if(LIBCLC_OFFLOAD_LIB)
    get_filename_component(LIBCLC_OFFLOAD_LIBDIR "${LIBCLC_OFFLOAD_LIB}" DIRECTORY)
  else()
    set(LIBCLC_TEST_ARCHITECTURE "")
    message(STATUS "libclc execution tests disabled: offload runtime not found")
  endif()
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-libclc "Running libclc tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS libclc
)
set_target_properties(check-libclc PROPERTIES FOLDER "libclc tests")
