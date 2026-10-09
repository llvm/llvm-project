if (NOT WIN32 OR NOT LLVM_LINK_LLVM_DYLIB)
  llvm_add_library(
      ClangdFeatureModuleExample
      MODULE FeatureModuleExample.cpp
      PLUGIN_TOOL clangd
      PARTIAL_SOURCES_INTENDED)
  llvm_add_library(
      ClangdStandaloneTweakExample
      MODULE StandaloneTweakExample.cpp
      PLUGIN_TOOL clangd
      PARTIAL_SOURCES_INTENDED)
endif()

foreach(target ClangdFeatureModuleExample ClangdStandaloneTweakExample)
  if(TARGET ${target})
    list(APPEND CLANGD_TEST_DEPS ${target})
    target_include_directories(${target} PUBLIC BEFORE
      "${CLANG_TOOLS_SOURCE_DIR}/clangd"
      "${CMAKE_CURRENT_BINARY_DIR}/../.."
    )
    if(CLANG_PLUGIN_SUPPORT AND (WIN32 OR CYGWIN))
      set(LLVM_LINK_COMPONENTS
        Support
      )
    endif()
  endif()
endforeach()

set(CLANGD_TEST_DEPS ${CLANGD_TEST_DEPS} PARENT_SCOPE)
