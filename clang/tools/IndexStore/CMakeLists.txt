include(CheckIncludeFiles)

set(SOURCES
  IndexStore.cpp

  ADDITIONAL_HEADERS
  ../../include/indexstore/indexstore.h
  ../../include/indexstore/IndexStoreCXX.h
  )

set(LIBS
  clangDirectoryWatcher
  clangIndex
)

set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/IndexStore.exports)

set(ENABLE_SHARED SHARED)

if(WIN32)
  set(output_name "libIndexStore")
else()
  set(output_name "IndexStore")
endif()

# FIXME: needs to be ported to non-Apple platforms.
if(APPLE)

add_clang_library(IndexStore ${ENABLE_SHARED} ${ENABLE_STATIC}
  OUTPUT_NAME ${output_name}
  ${SOURCES}

  LINK_LIBS
  ${LIBS}

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  Support
  )

set(INDEXSTORE_LIBRARY_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")

if(ENABLE_SHARED)
  if(WIN32)
    set_target_properties(IndexStore
      PROPERTIES
      VERSION ${INDEXSTORE_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
  elseif(APPLE)
    set(INDEXSTORE_LINK_FLAGS " -Wl,-compatibility_version -Wl,1")
    set(INDEXSTORE_LINK_FLAGS "${INDEXSTORE_LINK_FLAGS} -Wl,-current_version -Wl,${INDEXSTORE_LIBRARY_VERSION}")

    check_include_files("CoreServices/CoreServices.h" HAVE_CORESERVICES_H)
    if(HAVE_CORESERVICES_H)
      set(INDEXSTORE_LINK_FLAGS "${INDEXSTORE_LINK_FLAGS} -framework CoreServices")
    endif()

    set_property(TARGET IndexStore APPEND_STRING PROPERTY
                 LINK_FLAGS ${INDEXSTORE_LINK_FLAGS})
  else()
    set_target_properties(IndexStore
      PROPERTIES
      VERSION ${INDEXSTORE_LIBRARY_VERSION}
      DEFINE_SYMBOL _CINDEX_LIB_)
  endif()
endif()

if (LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(TARGETS IndexStore
    COMPONENT IndexStore
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION bin)

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-IndexStore
                      DEPENDS IndexStore
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=IndexStore
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
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
endif()
