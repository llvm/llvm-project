# Building libaiir-cpp.so fails if LLVM_ENABLE_PIC=Off
if (NOT LLVM_ENABLE_PIC)
  return()
endif()

# Building libaiir-cpp.so may not work on MSVC
if (MSVC)
  return()
endif()

get_property(aiir_libs GLOBAL PROPERTY AIIR_STATIC_LIBS)
get_property(aiir_llvm_link_components GLOBAL PROPERTY AIIR_LLVM_LINK_COMPONENTS)
list(REMOVE_DUPLICATES aiir_libs)
list(REMOVE_DUPLICATES aiir_llvm_link_components)

foreach (lib ${aiir_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  # libClang needs this, but it causes problems for AIIR (probably
  # because we use public library dependencies within AIIR.)
  # list(APPEND _DEPS  $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach ()

if(AIIR_LINK_AIIR_DYLIB)
  set(INSTALL_WITH_TOOLCHAIN INSTALL_WITH_TOOLCHAIN)
endif()

if(LLVM_BUILD_LLVM_DYLIB)
  add_aiir_library(
    AIIR
    SHARED
    EXCLUDE_FROM_LIBAIIR
    ${INSTALL_WITH_TOOLCHAIN}
    aiir-shlib.cpp
    ${_OBJECTS}
    LINK_LIBS
    ${_DEPS}

    LINK_COMPONENTS
    ${aiir_llvm_link_components}
  )
  target_link_libraries(AIIR PRIVATE ${LLVM_PTHREAD_LIB})
endif()

#message("Libraries included in libAIIR.so: ${aiir_libs}")
#message("LLVM Components included in libAIIR.so: ${aiir_llvm_link_components}")

aiir_check_all_link_libraries(AIIR)
