if(CLANG_PLUGIN_SUPPORT AND LLVM_ENABLE_PLUGINS AND NOT WIN32)
  add_subdirectory(TestTransformationPlugin)
endif()
