add_executable(sycl-ls sycl-ls.cpp)

target_include_directories(sycl-ls SYSTEM PRIVATE ${LLVM_MAIN_INCLUDE_DIR})
target_link_libraries(sycl-ls PRIVATE LLVMSupport LLVMObject)

add_dependencies(sycl-ls sycl)
target_include_directories(sycl-ls PRIVATE ${LIBSYCL_BUILD_INCLUDE_DIR})

target_link_libraries(sycl-ls
  PRIVATE
    ${LIBSYCL_SHARED_OUTPUT_NAME}
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fno-rtti COMPILER_HAS_NORTTI_FLAG)
if (NOT LLVM_ENABLE_RTTI AND COMPILER_HAS_NORTTI_FLAG)
  target_compile_options(sycl-ls PRIVATE -fno-rtti)
endif()

if (WIN32)
  # For security purposes, searches for dependent libraries are limited to
  # the combination of directories specified by the following flags of
  # LoadLibraryEx().
  # - LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR (0x100)
  # - LOAD_LIBRARY_SEARCH_SYSTEM32     (0x800)
  target_link_options(sycl-ls PRIVATE LINKER:/DEPENDENTLOADFLAG:0x900)
endif()
install(TARGETS sycl-ls
  RUNTIME DESTINATION "bin" COMPONENT sycl-ls)
