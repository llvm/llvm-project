# Building libclang-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  return()
endif()

get_property(clang_libs GLOBAL PROPERTY CLANG_STATIC_LIBS)

foreach (lib ${clang_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  if (BUILD_SHARED_LIBS)
    # If we are building static libraries, then we don't need to add the static
    # libraries as a dependency, because we are already linking against the
    # individual object files.
    list(APPEND _DEPS $<TARGET_PROPERTY:${lib},INTERFACE_LINK_LIBRARIES>)
  endif()

  # clang libraries are redundant since we are linking all the individual
  # object files into libclang-cpp.so, so filter them out from _DEPS.
  # This avoids problems with LLVM global data when building with
  # BUILD_SHARED_LIBS=ON
  # FIXME: We could use list(FILTER) with cmake >= 3.6
  # FIXME: With cmake >= 3.15 we could use the generator expression
  # $<FILTER:list,INCLUDE|EXCLUDE,regex>
  get_target_property(interface ${lib} LINK_LIBRARIES)
  if (interface)
    foreach(lib ${interface})
      if (NOT ${lib} MATCHES "^clang")
        list(APPEND _DEPS ${lib})
      endif()
    endforeach()
  endif()
endforeach ()

if (CLANG_LINK_CLANG_DYLIB)
  set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
endif()

if (HAIKU)
  list(APPEND _DEPS network)
endif()

add_clang_library(clang-cpp
                  SHARED
                  ${INSTALL_WITH_TOOLCHAIN}
                  clang-shlib.cpp
                  ${_OBJECTS}
                  LINK_LIBS
                  ${_DEPS})

configure_file(simple_version_script.map.in simple_version_script.map)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(clang-cpp PRIVATE LINKER:--version-script,${CMAKE_CURRENT_BINARY_DIR}/simple_version_script.map)

  # MLIR is statically linked into libclang-cpp.so (by ClangIR) but is also
  # provided by libMLIR.so in dylib builds.  Do not export anything that comes
  # from the MLIR archives: the dynamic linker could otherwise bind another
  # DSO's references (such as the vtable of an MLIR pattern defined outside of
  # an anonymous namespace) to clang-cpp's private copy, which is not
  # interchangeable with libMLIR.so's (e.g. TypeIDs differ).  Unlike the
  # version script, this doesn't depend on the namespace a symbol is in.
  set(_mlir_archives)
  foreach(dep ${_DEPS})
    if(TARGET ${dep} AND "${dep}" MATCHES "^MLIR")
      list(APPEND _mlir_archives $<TARGET_FILE_NAME:${dep}>)
    endif()
  endforeach()
  if(_mlir_archives)
    list(JOIN _mlir_archives "," _mlir_archives_arg)
    # -Xlinker keeps the comma-separated list in a single argument.
    target_link_options(clang-cpp PRIVATE
                        "SHELL:-Xlinker --exclude-libs=${_mlir_archives_arg}")
  endif()
endif()

# Optimize function calls for default visibility definitions to avoid PLT and
# reduce dynamic relocations.
if (NOT APPLE AND LLVM_LINKER_SUPPORTS_B_SYMBOLIC_FUNCTIONS)
  target_link_options(clang-cpp PRIVATE LINKER:-Bsymbolic-functions)
endif()
if (MINGW OR CYGWIN)
  # The clang-cpp DLL is supposed to export all symbols (except for ones
  # that are explicitly hidden). Normally, this is what happens anyway, but
  # if there are symbols that are marked explicitly as dllexport, we'd only
  # export them and nothing else. Therefore, add --export-all-symbols to
  # make sure we export all symbols despite potential dllexports.
  target_link_options(clang-cpp PRIVATE LINKER:--export-all-symbols)
endif()
