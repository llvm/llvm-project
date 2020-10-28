file(GLOB SWIG_INTERFACES interface/*.i)
file(GLOB_RECURSE SWIG_SOURCES *.swig)
file(GLOB SWIG_HEADERS
  ${LLDB_SOURCE_DIR}/include/lldb/API/*.h
  ${LLDB_SOURCE_DIR}/include/lldb/*.h
)
file(GLOB SWIG_PRIVATE_HEADERS
  ${LLDB_SOURCE_DIR}/include/lldb/lldb-private*.h
)
foreach(private_header ${SWIG_PRIVATE_HEADERS})
  list(REMOVE_ITEM SWIG_HEADERS ${private_header})
endforeach()

if(LLDB_BUILD_FRAMEWORK)
  set(framework_arg --framework --target-platform Darwin)
endif()

if(APPLE)
  set(DARWIN_EXTRAS "-D__APPLE__")
else()
  set(DARWIN_EXTRAS "")
endif()

set(SWIG_COMMON_FLAGS
  -c++
  -features autodoc
  -I${LLDB_SOURCE_DIR}/include
  -I${CMAKE_CURRENT_SOURCE_DIR}
  -D__STDC_LIMIT_MACROS
  -D__STDC_CONSTANT_MACROS
  ${DARWIN_EXTRAS}
  -outdir ${CMAKE_CURRENT_BINARY_DIR}
)

if (LLDB_ENABLE_PYTHON)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
    DEPENDS ${SWIG_SOURCES}
    DEPENDS ${SWIG_INTERFACES}
    DEPENDS ${SWIG_HEADERS}
    COMMAND ${SWIG_EXECUTABLE}
        ${SWIG_COMMON_FLAGS}
        -c++
        -shadow
        -python
        -threads
        -o ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
        ${LLDB_SOURCE_DIR}/bindings/python.swig
    VERBATIM
    COMMENT "Building LLDB Python wrapper")

  add_custom_target(swig_wrapper ALL DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapPython.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/lldb.py
  )
endif()

if (LLDB_ENABLE_LUA)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
    DEPENDS ${SWIG_SOURCES}
    DEPENDS ${SWIG_INTERFACES}
    DEPENDS ${SWIG_HEADERS}
    COMMAND ${SWIG_EXECUTABLE}
        ${SWIG_COMMON_FLAGS}
        -lua
        -w503
        -o ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
        ${LLDB_SOURCE_DIR}/bindings/lua.swig
    VERBATIM
    COMMENT "Building LLDB Lua wrapper")

  add_custom_target(swig_wrapper_lua ALL DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/LLDBWrapLua.cpp
  )
endif()

function(create_python_package swig_target working_dir pkg_dir)
  cmake_parse_arguments(ARG "NOINIT" "" "FILES" ${ARGN})
  if(ARG_FILES)
    set(copy_cmd COMMAND ${CMAKE_COMMAND} -E copy ${ARG_FILES} ${pkg_dir})
  endif()
  if(NOT ARG_NOINIT)
    set(init_cmd COMMAND ${PYTHON_EXECUTABLE}
        "${LLDB_SOURCE_DIR}/bindings/python/createPythonInit.py"
        "${pkg_dir}" ${ARG_FILES})
  endif()
  add_custom_command(TARGET ${swig_target} POST_BUILD VERBATIM
    COMMAND ${CMAKE_COMMAND} -E make_directory ${pkg_dir}
    ${copy_cmd}
    ${init_cmd}
    WORKING_DIRECTORY ${working_dir})
endfunction()

function(create_relative_symlink swig_target dest_file output_dir output_name)
  get_filename_component(dest_file ${dest_file} ABSOLUTE)
  get_filename_component(output_dir ${output_dir} ABSOLUTE)
  file(RELATIVE_PATH rel_dest_file ${output_dir} ${dest_file})
  if(CMAKE_HOST_UNIX)
    set(LLVM_LINK_OR_COPY create_symlink)
  else()
    set(LLVM_LINK_OR_COPY copy)
  endif()
  add_custom_command(TARGET ${swig_target} POST_BUILD VERBATIM
    COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} ${rel_dest_file} ${output_name}
    WORKING_DIRECTORY ${output_dir})
endfunction()

function(finish_swig swig_target lldb_bindings_dir lldb_python_target_dir)
  # Add a Post-Build Event to copy over Python files and create the symlink to
  # liblldb.so for the Python API(hardlink on Windows).
  add_custom_target(${swig_target} ALL VERBATIM
    COMMAND ${CMAKE_COMMAND} -E make_directory ${lldb_python_target_dir}
    DEPENDS ${lldb_bindings_dir}/lldb.py
    COMMENT "Python script sym-linking LLDB Python API")

  if(NOT LLDB_USE_SYSTEM_SIX)
    add_custom_command(TARGET ${swig_target} POST_BUILD VERBATIM
      COMMAND ${CMAKE_COMMAND} -E copy
        "${LLDB_SOURCE_DIR}/third_party/Python/module/six/six.py"
        "${lldb_python_target_dir}/../six.py")
  endif()

  add_custom_command(TARGET ${swig_target} POST_BUILD VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy
      "${lldb_bindings_dir}/lldb.py"
      "${lldb_python_target_dir}/__init__.py")

  add_custom_command(TARGET ${swig_target} POST_BUILD VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy
      "${LLDB_SOURCE_DIR}/source/Interpreter/embedded_interpreter.py"
      "${lldb_python_target_dir}")

  # Distribute the examples as python packages.
  create_python_package(
    ${swig_target}
    ${lldb_python_target_dir}
    "formatters/cpp"
    FILES "${LLDB_SOURCE_DIR}/examples/synthetic/gnu_libstdcpp.py"
          "${LLDB_SOURCE_DIR}/examples/synthetic/libcxx.py")

  create_python_package(
    ${swig_target}
    ${lldb_python_target_dir}
    "formatters"
    FILES "${LLDB_SOURCE_DIR}/examples/summaries/cocoa/cache.py"
          "${LLDB_SOURCE_DIR}/examples/summaries/synth.py"
          "${LLDB_SOURCE_DIR}/examples/summaries/cocoa/metrics.py"
          "${LLDB_SOURCE_DIR}/examples/summaries/cocoa/attrib_fromdict.py"
          "${LLDB_SOURCE_DIR}/examples/summaries/cocoa/Logger.py")

  create_python_package(
    ${swig_target}
    ${lldb_python_target_dir}
    "utils"
    FILES "${LLDB_SOURCE_DIR}/examples/python/in_call_stack.py"
          "${LLDB_SOURCE_DIR}/examples/python/symbolication.py")

  if(APPLE)
    create_python_package(
      ${swig_target}
      ${lldb_python_target_dir} "macosx"
      FILES "${LLDB_SOURCE_DIR}/examples/python/crashlog.py"
            "${LLDB_SOURCE_DIR}/examples/darwin/heap_find/heap.py")

    create_python_package(
      ${swig_target}
      ${lldb_python_target_dir} "macosx/heap"
      FILES "${LLDB_SOURCE_DIR}/examples/darwin/heap_find/heap/heap_find.cpp"
            "${LLDB_SOURCE_DIR}/examples/darwin/heap_find/heap/Makefile"
            NOINIT)

    create_python_package(
      ${swig_target}
      ${lldb_python_target_dir} "diagnose"
      FILES "${LLDB_SOURCE_DIR}/examples/python/diagnose_unwind.py"
            "${LLDB_SOURCE_DIR}/examples/python/diagnose_nsstring.py")
  endif()

  if(LLDB_BUILD_FRAMEWORK)
    set(LIBLLDB_SYMLINK_DEST "${LLDB_FRAMEWORK_ABSOLUTE_BUILD_DIR}/LLDB.framework/LLDB")
  else()
    set(LIBLLDB_SYMLINK_DEST "${LLVM_SHLIB_OUTPUT_INTDIR}/liblldb${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  if(WIN32)
    if(CMAKE_BUILD_TYPE STREQUAL Debug)
      set(LIBLLDB_SYMLINK_OUTPUT_FILE "_lldb_d.pyd")
    else()
      set(LIBLLDB_SYMLINK_OUTPUT_FILE "_lldb.pyd")
    endif()
  else()
    set(LIBLLDB_SYMLINK_OUTPUT_FILE "_lldb.so")
  endif()
  create_relative_symlink(${swig_target} ${LIBLLDB_SYMLINK_DEST}
                          ${lldb_python_target_dir} ${LIBLLDB_SYMLINK_OUTPUT_FILE})

  if(NOT LLDB_BUILD_FRAMEWORK)
    set(LLDB_ARGDUMPER_FILENAME "lldb-argdumper${CMAKE_EXECUTABLE_SUFFIX}")
    create_relative_symlink(${swig_target} "${LLVM_RUNTIME_OUTPUT_INTDIR}/${LLDB_ARGDUMPER_FILENAME}"
                            ${lldb_python_target_dir} ${LLDB_ARGDUMPER_FILENAME})
  endif()

  add_dependencies(${swig_target} swig_wrapper liblldb lldb-argdumper)
  set_target_properties(${swig_target} swig_wrapper PROPERTIES FOLDER "lldb misc")

  # Ensure we do the python post-build step when building lldb.
  add_dependencies(lldb ${swig_target})

  # Install the LLDB python module
  if(LLDB_BUILD_FRAMEWORK)
    set(LLDB_PYTHON_INSTALL_PATH ${LLDB_FRAMEWORK_INSTALL_DIR}/LLDB.framework/Resources/Python)
  else()
    set(LLDB_PYTHON_INSTALL_PATH ${LLDB_PYTHON_RELATIVE_PATH})
  endif()
  if (NOT CMAKE_CFG_INTDIR STREQUAL  ".")
    string(REPLACE ${CMAKE_CFG_INTDIR} "\$\{CMAKE_INSTALL_CONFIG_NAME\}" LLDB_PYTHON_INSTALL_PATH ${LLDB_PYTHON_INSTALL_PATH})
    string(REPLACE ${CMAKE_CFG_INTDIR} "\$\{CMAKE_INSTALL_CONFIG_NAME\}" lldb_python_target_dir ${lldb_python_target_dir})
  endif()
  set(swig_scripts_target "${swig_target}_scripts")
  set(swig_scripts_install_target "${swig_target}_scripts_install")
  add_custom_target(${swig_scripts_target})
  add_dependencies(${swig_scripts_target} ${swig_target})
  install(DIRECTORY ${lldb_python_target_dir}/../
          DESTINATION ${LLDB_PYTHON_INSTALL_PATH}
          COMPONENT ${swig_scripts_target})
  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(${swig_scripts_install_target}
                             COMPONENT ${swig_scripts_target}
                             DEPENDS ${swig_scripts_target})
  endif()

  # Add a Post-Build Event to copy the custom Python DLL to the lldb binaries dir so that Windows can find it when launching
  # lldb.exe or any other executables that were linked with liblldb.
  if (WIN32 AND NOT "${PYTHON_DLL}" STREQUAL "")
    # When using the Visual Studio CMake generator the lldb binaries end up in Release/bin, Debug/bin etc.
    file(TO_NATIVE_PATH "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin" LLDB_BIN_DIR)
    file(TO_NATIVE_PATH "${PYTHON_DLL}" PYTHON_DLL_NATIVE_PATH)
    add_custom_command(
      TARGET ${swig_target}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${PYTHON_DLL_NATIVE_PATH} ${LLDB_BIN_DIR} VERBATIM
      COMMENT "Copying Python DLL to LLDB binaries directory.")
  endif()
endfunction()
