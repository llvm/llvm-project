include(CheckIncludeFiles)

set(SOURCES
  IndexStore.cpp

  ADDITIONAL_HEADERS
  ../../include/indexstore/indexstore.h
  ../../include/indexstore/IndexStoreCXX.h
  )

set(LIBS
  clangIndex
  clangIndexDataStore
)

if(NOT CMAKE_SYSTEM_NAME STREQUAL Windows)
  set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/IndexStore.exports)
endif()

if(WIN32)
  set(output_name "libIndexStore")
else()
  set(output_name "IndexStore")
endif()

add_clang_library(IndexStore SHARED
  OUTPUT_NAME ${output_name}
  ${SOURCES}

  LINK_LIBS
  ${LIBS}

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  Support
  )

if(APPLE)
  set(INDEXSTORE_LIBRARY_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")

  set(INDEXSTORE_LINK_FLAGS " -Wl,-compatibility_version -Wl,1")
  set(INDEXSTORE_LINK_FLAGS "${INDEXSTORE_LINK_FLAGS} -Wl,-current_version -Wl,${INDEXSTORE_LIBRARY_VERSION}")

  check_include_files("CoreServices/CoreServices.h" HAVE_CORESERVICES_H)
  if(HAVE_CORESERVICES_H)
    set(INDEXSTORE_LINK_FLAGS "${INDEXSTORE_LINK_FLAGS} -framework CoreServices")
  endif()

  set_property(TARGET IndexStore APPEND_STRING PROPERTY
               LINK_FLAGS ${INDEXSTORE_LINK_FLAGS})
endif()

if (LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(TARGETS IndexStore
    COMPONENT IndexStore
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION bin)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_llvm_install_targets(install-IndexStore
                             DEPENDS IndexStore
                             COMPONENT IndexStore)
  endif()
endif()

set(INDEXSTORE_HEADERS_INSTALL_DESTINATION "local/include")

install(DIRECTORY ../../include/indexstore
  COMPONENT IndexStore
  DESTINATION "${INDEXSTORE_HEADERS_INSTALL_DESTINATION}"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
  )
