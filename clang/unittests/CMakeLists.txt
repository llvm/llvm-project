add_custom_target(ClangUnitTests)
set_target_properties(ClangUnitTests PROPERTIES FOLDER "Clang/Tests")

if(CLANG_BUILT_STANDALONE)
  # LLVMTesting* libraries are needed for some of the unittests.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Testing/Annotations
      AND NOT TARGET LLVMTestingAnnotations)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Testing/Annotations
      lib/Testing/Annotations)
  endif()
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      AND NOT TARGET LLVMTestingSupport)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      lib/Testing/Support)
  endif()
endif()

# add_distinct_clang_unittest(test_name file1.cpp file2.cpp)
#
# Will compile the list of files together and link against the clang
# Produces a binary named 'basename(test_name)'.
function(add_distinct_clang_unittest test_name)
  cmake_parse_arguments(ARG
    ""
    ""
    "CLANG_LIBS;LINK_LIBS;LLVM_COMPONENTS"
    ${ARGN})

  if (NOT ${test_name} MATCHES "Tests$")
    message(FATAL_ERROR "Unit test name must end with 'Tests' for lit to find it.")
  endif()

  # LLVM_COMPONENTS is for LLVM_LINK_COMPONENTS deps, and must be before
  # add_unittest.
  list(APPEND LLVM_LINK_COMPONENTS ${ARG_LLVM_COMPONENTS})

  add_unittest(ClangUnitTests ${test_name} ${ARG_UNPARSED_ARGUMENTS})

  # Clang libs either come from the entire dylib, or individual libraries.
  if (CLANG_LINK_CLANG_DYLIB)
    list(APPEND ARG_LINK_LIBS clang-cpp)
  else()
    list(APPEND ARG_LINK_LIBS ${ARG_CLANG_LIBS})
  endif()

  # LINK_LIBS is for normal library dependencies.
  target_link_libraries(${test_name} PRIVATE ${ARG_LINK_LIBS})
endfunction()

set(doc_opts BRIEF_DOCS "<internal setting>" FULL_DOCS "<internal settings>")
define_property(GLOBAL PROPERTY CLANG_UNITTEST_SRCS ${doc_opts})
define_property(GLOBAL PROPERTY CLANG_UNITTEST_LLVM_COMPONENTS ${doc_opts})
define_property(GLOBAL PROPERTY CLANG_UNITTEST_CLANG_LIBS ${doc_opts})
define_property(GLOBAL PROPERTY CLANG_UNITTEST_LINK_LIBS ${doc_opts})

# add_clang_unittest(test_name file1.cpp file2.cpp)
#
# Adds unittests to the combined AllClangUnitTests binary. The unittest binary
# is defined after adding all unittest subdirectories.
function(add_clang_unittest test_name)
  cmake_parse_arguments(ARG
    ""
    ""
    "CLANG_LIBS;LINK_LIBS;LLVM_COMPONENTS"
    ${ARGN})

  file(RELATIVE_PATH src_prefix "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  set(srcs_prefixed)
  foreach(src ${ARG_UNPARSED_ARGUMENTS})
    set(srcs_prefixed ${srcs_prefixed} "${src_prefix}/${src}")
  endforeach()
  set_property(GLOBAL APPEND PROPERTY CLANG_UNITTEST_SRCS ${srcs_prefixed})
  set_property(GLOBAL APPEND PROPERTY CLANG_UNITTEST_CLANG_LIBS ${ARG_CLANG_LIBS})
  set_property(GLOBAL APPEND PROPERTY CLANG_UNITTEST_LINK_LIBS ${ARG_LINK_LIBS})
  set_property(GLOBAL APPEND PROPERTY CLANG_UNITTEST_LLVM_COMPONENTS ${ARG_LLVM_COMPONENTS})
endfunction()

add_subdirectory(Basic)
add_subdirectory(Lex)
add_subdirectory(Parse)
add_subdirectory(Driver)
if(CLANG_ENABLE_STATIC_ANALYZER)
  add_subdirectory(Analysis)
  add_subdirectory(StaticAnalyzer)
endif()
add_subdirectory(ASTMatchers)
add_subdirectory(AST)
add_subdirectory(CrossTU)
add_subdirectory(Tooling)
add_subdirectory(Format)
add_subdirectory(Frontend)
add_subdirectory(Rewrite)
add_subdirectory(Sema)
add_subdirectory(CodeGen)
if(HAVE_CLANG_REPL_SUPPORT)
  add_subdirectory(Interpreter)
endif()
# FIXME: libclang unit tests are disabled on Windows due
# to failures, mostly in libclang.VirtualFileOverlay_*.
if(NOT WIN32 AND CLANG_TOOL_LIBCLANG_BUILD) 
  add_subdirectory(libclang)
endif()
add_subdirectory(DirectoryWatcher)
add_subdirectory(Index)
add_subdirectory(InstallAPI)
add_subdirectory(Serialization)
add_subdirectory(Support)
if (CLANG_ENABLE_CIR)
  add_subdirectory(CIR)
endif()

# If we're doing a single merged clang unit test binary, add that target after
# all the previous subdirectories have been processed.
get_property(SRCS GLOBAL PROPERTY CLANG_UNITTEST_SRCS)
get_property(CLANG_LIBS GLOBAL PROPERTY CLANG_UNITTEST_CLANG_LIBS)
get_property(LINK_LIBS GLOBAL PROPERTY CLANG_UNITTEST_LINK_LIBS)
get_property(LLVM_COMPONENTS GLOBAL PROPERTY CLANG_UNITTEST_LLVM_COMPONENTS)
add_distinct_clang_unittest(AllClangUnitTests
  ${SRCS}
  AllClangUnitTests.cpp
  CLANG_LIBS
  ${CLANG_LIBS}
  LINK_LIBS
  ${LINK_LIBS}
  LLVM_COMPONENTS
  ${LLVM_COMPONENTS}
)

# The Tooling library has some internal headers. Make those work. If we like
# the merged clang unit test binary, we can update the include paths and make
# this the default.
include_directories(Tooling)
