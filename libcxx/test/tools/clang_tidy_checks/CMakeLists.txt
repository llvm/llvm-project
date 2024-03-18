
# The find_package changes these variables. This leaves the build in an odd
# state. Calling cmake a second time tries to write site config information in
# the system's libc++. Restoring these setting after testing fixes this issue.
set(LLVM_DIR_SAVE ${LLVM_DIR})
set(Clang_DIR_SAVE ${Clang_DIR})

# Since the Clang C++ ABI is not stable the Clang libraries and clang-tidy
# versions must match. Otherwise there likely will be ODR-violations. This had
# led to crashes and incorrect output of the clang-tidy based checks.
find_package(Clang ${CMAKE_CXX_COMPILER_VERSION})
if(NOT Clang_FOUND)
  message(STATUS "Clang-tidy tests are disabled since the "
                 "Clang development package is unavailable.")
  return()
endif()
if(NOT TARGET clangTidy)
  message(STATUS "Clang-tidy tests are disabled since the "
                 "Clang development package has no clangTidy target.")
  return()
endif()
message(STATUS "Clang-tidy tests are enabled.")

set(SOURCES
    abi_tag_on_virtual.cpp
    header_exportable_declarations.cpp
    hide_from_abi.cpp
    proper_version_checks.cpp
    qualify_declval.cpp
    robust_against_adl.cpp
    uglify_attributes.cpp

    libcpp_module.cpp
   )

list(APPEND LIBCXX_TEST_DEPS cxx-tidy)

set(LLVM_DIR "${LLVM_DIR_SAVE}" CACHE PATH "The directory containing a CMake configuration file for LLVM." FORCE)
set(Clang_DIR "${Clang_DIR_SAVE}" CACHE PATH "The directory containing a CMake configuration file for Clang." FORCE)

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
target_compile_options(clangTidy INTERFACE
                       -fno-rtti
                       -fno-sanitize=address,hwaddress,undefined,thread,leak # ignore any sanitizers
                      )

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(clangTidy INTERFACE
                         -fno-sanitize=memory,dataflow
                        )
endif()

add_library(cxx-tidy MODULE ${SOURCES})
target_link_libraries(cxx-tidy clangTidy)

set_target_properties(cxx-tidy PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)

set_target_properties(cxx-tidy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_SHARED_MODULE_SUFFIX_CXX .plugin) # Use a portable suffix to simplify how we can find it from Lit
