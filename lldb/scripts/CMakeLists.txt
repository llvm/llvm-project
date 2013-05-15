set(LLVM_NO_RTTI 1)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
  DEPENDS ${LLDB_SOURCE_DIR}/scripts/lldb.swig
  # swig was directly invoked on Windows (where the Python API is not being generated) but on other platforms, we need to run the *swig-wrapper-classes.sh shell-scripts.
  #COMMAND swig -c++ -shadow -python -I${LLDB_SOURCE_DIR}/include -I./. -outdir ${LLDB_SOURCE_DIR}/scripts/Python  -o ${LLDB_SOURCE_DIR}/source/LLDBWrapPython.cpp ${LLDB_SOURCE_DIR}/scripts/lldb.swig
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build-swig-wrapper-classes.sh ${LLDB_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR} -m
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/finish-swig-wrapper-classes.sh ${LLDB_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR} -m
  COMMENT "Building lldb python wrapper")
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp PROPERTIES GENERATED 1)

ADD_CUSTOM_TARGET(swig_wrapper ALL echo -n
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
    )
