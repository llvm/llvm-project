set(SOURCES
  ARCMigrate.cpp
  BuildSystem.cpp
  CIndex.cpp
  CIndexCXX.cpp
  CIndexCodeCompletion.cpp
  CIndexDiagnostic.cpp
  CIndexHigh.cpp
  CIndexInclusionStack.cpp
  CIndexUSRs.cpp
  CIndexer.cpp
  CXComment.cpp
  CXCursor.cpp
  CXIndexDataConsumer.cpp
  CXCompilationDatabase.cpp
  CXLoadedDiagnostic.cpp
  CXSourceLocation.cpp
  CXStoredDiagnostic.cpp
  CXString.cpp
  CXType.cpp
  Indexing.cpp
  FatalErrorHandler.cpp

  ADDITIONAL_HEADERS
  CIndexDiagnostic.h
  CIndexer.h
  CXCursor.h
  CXLoadedDiagnostic.h
  CXSourceLocation.h
  CXString.h
  CXTranslationUnit.h
  CXType.h
  Index_Internal.h
  ../../include/clang-c/Index.h
  )

set(LIBS
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangIndex
  clangLex
  clangSema
  clangSerialization
  clangTooling
)

if (CLANG_ENABLE_ARCMT)
  list(APPEND LIBS clangARCMigrate)
endif ()

option(LIBCLANG_INCLUDE_CLANG_TOOLS_EXTRA
  "Include code from clang-tools-extra in libclang." OFF)

if (LIBCLANG_INCLUDE_CLANG_TOOLS_EXTRA)
  if (TARGET clangTidyPlugin)
    add_definitions(-DCLANG_TOOL_EXTRA_BUILD)
    list(APPEND LIBS clangTidyPlugin)
    list(APPEND LIBS clangIncludeFixerPlugin)
    if(LLVM_ENABLE_MODULES)
      list(APPEND LLVM_COMPILE_FLAGS "-fmodules-ignore-macro=CLANG_TOOL_EXTRA_BUILD")
    endif()
  else ()
    message(FATAL_ERROR "LIBCLANG_INCLUDE_CLANG_TOOLS_EXTRA needs clang-tools-extra in LLVM_ENABLE_PROJECTS")
  endif ()
endif ()

if (HAVE_LIBDL)
  list(APPEND LIBS ${CMAKE_DL_LIBS})
elseif (CLANG_BUILT_STANDALONE)
  find_library(DL_LIBRARY_PATH dl)
  if (DL_LIBRARY_PATH)
    list(APPEND LIBS dl)
  endif ()
endif ()

option(LIBCLANG_BUILD_STATIC
  "Build libclang as a static library (in addition to a shared one)" OFF)

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/libclang.exports)

if(MSVC)
  # Avoid LNK4197 by not specifying libclang.exports here.
  # Each functions is exported as "dllexport" in include/clang-c.
  # KB835326
  set(LLVM_EXPORTED_SYMBOL_FILE)
endif()

if(LLVM_ENABLE_PIC OR (WIN32 AND NOT LIBCLANG_BUILD_STATIC))
  set(ENABLE_SHARED SHARED)
endif()

if(NOT LLVM_ENABLE_PIC OR LIBCLANG_BUILD_STATIC)
  set(ENABLE_STATIC STATIC)
endif()

if (WIN32 AND ENABLE_SHARED AND ENABLE_STATIC)
  unset(ENABLE_STATIC)
endif()

if(WIN32)
  set(output_name "libclang")
else()
  set(output_name "clang")
endif()

# libclang requires headers which need _ALL_SOURCE to build on AIX
if (UNIX AND ${CMAKE_SYSTEM_NAME} MATCHES "AIX")
    remove_definitions("-D_XOPEN_SOURCE=700")
endif()

add_clang_library(libclang ${ENABLE_SHARED} ${ENABLE_STATIC} INSTALL_WITH_TOOLCHAIN
  OUTPUT_NAME ${output_name}
  ${SOURCES}

  DEPENDS
  ClangDriverOptions
  clang-resource-headers

  LINK_LIBS
  ${LIBS}

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  Support
  )

if(ENABLE_STATIC)
  foreach(name libclang obj.libclang libclang_static)
    if (TARGET ${name})
      target_compile_definitions(${name} PUBLIC CINDEX_NO_EXPORTS)
    endif()
  endforeach()
endif()

if(ENABLE_SHARED)
  if(WIN32)
    set_target_properties(libclang
      PROPERTIES
      VERSION ${LIBCLANG_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
  elseif(APPLE)
    set(LIBCLANG_LINK_FLAGS " -Wl,-compatibility_version -Wl,1")
    set(LIBCLANG_LINK_FLAGS "${LIBCLANG_LINK_FLAGS} -Wl,-current_version -Wl,${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")

    set_property(TARGET libclang APPEND_STRING PROPERTY
                 LINK_FLAGS ${LIBCLANG_LINK_FLAGS})
  else()
    set_target_properties(libclang
      PROPERTIES
      SOVERSION ${LIBCLANG_LIBRARY_VERSION}
      VERSION ${LIBCLANG_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
    # FIXME: _CINDEX_LIB_ affects dllexport/dllimport on Win32.
    if(LLVM_ENABLE_MODULES AND NOT WIN32)
      target_compile_options(libclang PRIVATE
        "-fmodules-ignore-macro=_CINDEX_LIB_"
        )
    endif()
  endif()
endif()

if(INTERNAL_INSTALL_PREFIX)
  set(LIBCLANG_HEADERS_INSTALL_DESTINATION "${INTERNAL_INSTALL_PREFIX}/include")
else()
  set(LIBCLANG_HEADERS_INSTALL_DESTINATION include)
endif()

install(DIRECTORY ../../include/clang-c
  COMPONENT libclang-headers
  DESTINATION "${LIBCLANG_HEADERS_INSTALL_DESTINATION}"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )

# LLVM_DISTRIBUTION_COMPONENTS requires that each component have both a
# component and an install-component target, so add a dummy libclang-headers
# target to allow using it in LLVM_DISTRIBUTION_COMPONENTS.
add_custom_target(libclang-headers)
set_target_properties(libclang-headers PROPERTIES FOLDER "Misc")

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-libclang-headers
                           COMPONENT libclang-headers)
endif()

# Create a target to install the python bindings to make them easier to
# distribute.  Since the bindings are over libclang, which is installed
# unbundled to the clang version, follow suit.
foreach(PythonVersion ${CLANG_PYTHON_BINDINGS_VERSIONS})
  install(DIRECTORY
            ${CMAKE_CURRENT_SOURCE_DIR}/../../bindings/python/clang
          COMPONENT
            libclang-python-bindings
          DESTINATION
            "lib${LLVM_LIBDIR_SUFFIX}/python${PythonVersion}/site-packages")
endforeach()
if(NOT LLVM_ENABLE_IDE)
  add_custom_target(libclang-python-bindings)
  add_llvm_install_targets(install-libclang-python-bindings
                           COMPONENT
                             libclang-python-bindings)
endif()

