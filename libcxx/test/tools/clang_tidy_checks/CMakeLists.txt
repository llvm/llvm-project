
# Look for the latest package
set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

find_package(Clang 16)

set(SOURCES
    abi_tag_on_virtual.cpp
    hide_from_abi.cpp
    qualify_declval.cpp
    robust_against_adl.cpp
    uglify_attributes.cpp

    libcpp_module.cpp
   )

if(NOT Clang_FOUND)
  message(STATUS "Could not find a suitable version of the Clang development package;
                  custom libc++ clang-tidy checks will not be available.")
  return()
endif()

message(STATUS "Found system-installed LLVM ${LLVM_PACKAGE_VERSION} with headers in ${LLVM_INCLUDE_DIRS}")

set(CMAKE_CXX_STANDARD 20)

# Link only against clangTidy itself, not anything that clangTidy uses; otherwise we run setup code multiple times
# which results in clang-tidy crashing
set_target_properties(clangTidy PROPERTIES INTERFACE_LINK_LIBRARIES "")
# ClangTargets.cmake doesn't set the include paths, so we have to do it
target_include_directories(clangTidy INTERFACE
                           ${CLANG_INCLUDE_DIRS}
                           ${LLVM_INCLUDE_DIRS}
                          )
target_compile_options(clangTidy INTERFACE -fno-rtti)

add_library(cxx-tidy MODULE ${SOURCES})
target_link_libraries(cxx-tidy clangTidy)

set_target_properties(cxx-tidy PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)

set_target_properties(cxx-tidy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_SHARED_MODULE_SUFFIX_CXX .plugin) # Use a portable suffix to simplify how we can find it from Lit
