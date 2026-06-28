# FindFlang.cmake

include(FindPackageHandleStandardArgs)

# If Flang and lldb are in-tree then the libraries will already be available, otherwise look for specific directories
if(TARGET flangFrontEnd)
  set(Flang_FOUND TRUE)
else()
  find_package(Flang QUIET CONFIG HINTS ${Flang_DIR} ${LLVM_DIR}/../flang)
endif()

find_package_handle_standard_args(Flang
  FOUND_VAR
    Flang_FOUND
  REQUIRED_VARS
    Flang_FOUND
)