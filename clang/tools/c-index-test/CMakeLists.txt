include(CheckIncludeFiles)

set(LLVM_LINK_COMPONENTS
  support
)

add_clang_executable(c-index-test
  c-index-test.c
  core_main.cpp
  JSONAggregation.cpp
  )

set(INDEXSTORE_LIB)
set(CINDEXTEST_LIBS)
if(APPLE)
  set(INDEXSTORE_LIB IndexStore)
endif()

if(NOT MSVC)
  set_property(
    SOURCE c-index-test.c
    PROPERTY COMPILE_FLAGS "-std=gnu89"
    )
endif()

if (LLVM_BUILD_STATIC)
  target_link_libraries(c-index-test
    libclang_static
    clangIndex
    ${CINDEXTEST_LIBS}
  )
else()
  target_link_libraries(c-index-test
    libclang
    ${INDEXSTORE_LIB}
    clangAST
    clangBasic
    clangCodeGen
    clangDirectoryWatcher
    clangFrontend
    clangIndex
    clangSerialization
    ${CINDEXTEST_LIBS}
  )
endif()

set_target_properties(c-index-test
  PROPERTIES
  LINKER_LANGUAGE CXX)

# If libxml2 is available, make it available for c-index-test.
if (CLANG_HAVE_LIBXML)
  include_directories(SYSTEM ${LIBXML2_INCLUDE_DIR})
  target_link_libraries(c-index-test ${LIBXML2_LIBRARIES})
endif()

if(APPLE)
  check_include_files("CoreServices/CoreServices.h" HAVE_CORESERVICES_H)
  if(HAVE_CORESERVICES_H)
    target_link_libraries(c-index-test "-framework CoreServices")
  endif()
endif()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  if(INTERNAL_INSTALL_PREFIX)
    set(INSTALL_DESTINATION "${INTERNAL_INSTALL_PREFIX}/bin")
    set_property(TARGET c-index-test APPEND PROPERTY INSTALL_RPATH
       "@executable_path/../../lib")
  else()
    set(INSTALL_DESTINATION bin)
  endif()

  install(TARGETS c-index-test
    RUNTIME DESTINATION "${INSTALL_DESTINATION}"
    COMPONENT c-index-test)

  if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
    add_custom_target(install-c-index-test
      DEPENDS c-index-test
      COMMAND "${CMAKE_COMMAND}"
              -DCMAKE_INSTALL_COMPONENT=c-index-test
              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
  endif()
endif()
