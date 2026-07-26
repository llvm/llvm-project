add_library(DefUseRuntime STATIC
  DefUseRuntime.cpp
)

set_target_properties(DefUseRuntime PROPERTIES
  OUTPUT_NAME DefUseRuntime
  ARCHIVE_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}"
  POSITION_INDEPENDENT_CODE ON
  FOLDER "Clang/Runtime"
)

# При сборке clang автоматически собираем и runtime.
if(TARGET clang)
  add_dependencies(clang DefUseRuntime)
endif()

install(TARGETS DefUseRuntime
  ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}"
  COMPONENT DefUseRuntime
)
