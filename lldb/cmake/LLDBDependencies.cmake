set(LLDB_SYSTEM_LIBS)

# Windows-only libraries
if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  list(APPEND LLDB_SYSTEM_LIBS
    ws2_32
    rpcrt4
    )
endif ()

if (NOT LLDB_DISABLE_LIBEDIT)
  list(APPEND LLDB_SYSTEM_LIBS edit)
endif()
if (NOT LLDB_DISABLE_CURSES)
  list(APPEND LLDB_SYSTEM_LIBS ${CURSES_LIBRARIES})
  if(LLVM_ENABLE_TERMINFO AND HAVE_TERMINFO)
    list(APPEND LLDB_SYSTEM_LIBS ${TERMINFO_LIBS})
  endif()
endif()

if (NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB )
    list(APPEND LLDB_SYSTEM_LIBS atomic)
endif()

list(APPEND LLDB_SYSTEM_LIBS ${Backtrace_LIBRARY})

if (NOT LLDB_DISABLE_PYTHON AND NOT LLVM_BUILD_STATIC)
  list(APPEND LLDB_SYSTEM_LIBS ${PYTHON_LIBRARIES})
endif()

list(APPEND LLDB_SYSTEM_LIBS ${system_libs})

if (LLVM_BUILD_STATIC)
  if (NOT LLDB_DISABLE_PYTHON)
    list(APPEND LLDB_SYSTEM_LIBS python2.7 util)
  endif()
  if (NOT LLDB_DISABLE_CURSES)
    list(APPEND LLDB_SYSTEM_LIBS gpm)
  endif()
endif()

if ( NOT LLDB_DISABLE_PYTHON )
  set_source_files_properties(${LLDB_WRAP_PYTHON} PROPERTIES GENERATED 1)
  if (CLANG_CL)
    set_source_files_properties(${LLDB_WRAP_PYTHON} PROPERTIES COMPILE_FLAGS -Wno-unused-function)
  endif()
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
      NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set_property(SOURCE ${LLDB_WRAP_PYTHON}
                 APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-sequence-point -Wno-cast-qual")
  endif ()
endif()
