# Building libclang_shared.so fails if LLVM_ENABLE_PIC=Off
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
  list(APPEND _DEPS $<TARGET_PROPERTY:${lib},INTERFACE_LINK_LIBRARIES>)
  list(APPEND _DEPS $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach ()

add_clang_library(clang_shared
                  SHARED
                  clang-shlib.cpp
                  ${_OBJECTS}
                  LINK_LIBS
                  ${_DEPS})
