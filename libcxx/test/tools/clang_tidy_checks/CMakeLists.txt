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

# In some cases even with the clangTidy target present the headers appear not to
# be on the system. Run a short test to see whether the header is present.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test.cpp" "
#if !__has_include(\"clang-tidy/ClangTidyCheck.h\")
  # error No clang-tidy headers
#endif
int main(){}
")
try_compile(HAS_CLANG_TIDY_HEADERS
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/test.cpp"
   LINK_LIBRARIES clangTidy)

if(NOT HAS_CLANG_TIDY_HEADERS)
  message(STATUS "Clang-tidy tests are disabled since the "
                 "clang-tidy headers are not present.")
  return()
endif()

# The clangTidy plugin uses C++20, so ensure that we support C++20 when using libstdc++.
# This is required because some versions of libstdc++ used as a system library on build platforms
# we support do not support C++20 yet.
# Note it has not been tested whether version 11 works.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/test.cpp" "
#include <version>
#if defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE < 11
  # error The libstdc++ version is too old.
#endif
int main(){}
")
try_compile(HAS_NEWER_STANDARD_LIBRARY
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/test.cpp"
   LINK_LIBRARIES clangTidy)

if(NOT HAS_NEWER_STANDARD_LIBRARY)
  message(STATUS "Clang-tidy tests are disabled due to using "
                 "stdlibc++ older than version 11")
  return()
endif()
message(STATUS "Clang-tidy tests are enabled.")

set(SOURCES
    abi_tag_on_virtual.cpp
    header_exportable_declarations.cpp
    hide_from_abi.cpp
    internal_ftm_use.cpp
    nodebug_on_aliases.cpp
    proper_version_checks.cpp
    qualify_declval.cpp
    robust_against_adl.cpp
    uglify_attributes.cpp

    libcpp_module.cpp
   )

add_library(cxx-tidy MODULE ${SOURCES})
target_link_libraries(cxx-tidy clangTidy)

set_target_properties(cxx-tidy PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENSIONS NO)

set_target_properties(cxx-tidy PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_SHARED_MODULE_SUFFIX_CXX .plugin) # Use a portable suffix to simplify how we can find it from Lit

add_dependencies(cxx-test-depends cxx-tidy)
