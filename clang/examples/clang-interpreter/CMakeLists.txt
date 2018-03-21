set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MC
  MCJIT
  OrcJit
  Option
  RuntimeDyld
  Support
  native
  )

add_clang_executable(clang-interpreter
  main.cpp
  Invoke.cpp
  Manager.cpp
  )

add_dependencies(clang-interpreter
  clang-headers
  )

target_link_libraries(clang-interpreter
  PRIVATE
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  )

export_executable_symbols(clang-interpreter)

if (MSVC)
  # Is this a CMake bug that even with export_executable_symbols, Windows
  # needs to explictly export the type_info vtable
  set_property(TARGET clang-interpreter
               APPEND_STRING PROPERTY LINK_FLAGS " /EXPORT:??_7type_info@@6B@")
endif()

function(clang_enable_exceptions TARGET)
  # Really have to jump through hoops to enable exception handling independent
  # of how LLVM is being built.
  if (NOT LLVM_REQUIRES_EH AND NOT LLVM_REQUIRES_RTTI)
    if (MSVC)
      # /EHs to allow throwing from extern "C"
      set(excptnExceptions_ON "/D _HAS_EXCEPTIONS=1 /EHs /wd4714")
      set(excptnExceptions_OFF "/D _HAS_EXCEPTIONS=0 /EHs-c-")
      set(excptnRTTI_ON "/GR")
      set(excptnRTTI_OFF "/GR-")
      set(excptnEHRTTIRegEx "(/EHs(-c-?)|_HAS_EXCEPTIONS=(0|1))")
    else()
      set(excptnExceptions_ON "-fexceptions")
      set(excptnExceptions_OFF "-fno-exceptions")
      set(excptnRTTI_ON "-frtti")
      set(excptnRTTI_OFF "-fno-rtti")
      set(excptnEHRTTIRegEx "-f(exceptions|no-exceptions)")
    endif()
    if (LLVM_REQUIRES_EH)
      set(excptnExceptions_DFLT ${excptnExceptions_ON})
    else()
      set(excptnExceptions_DFLT ${excptnExceptions_OFF})
    endif()
    if (LLVM_REQUIRES_RTTI)
      set(excptnRTTI_DFLT ${excptnRTTI_ON})
    else()
      set(excptnRTTI_DFLT ${excptnRTTI_OFF})
    endif()

    # Strip the exception & rtti flags from the target
    get_property(addedFlags TARGET ${TARGET} PROPERTY COMPILE_FLAGS)
    string(REGEX REPLACE ${excptnEHRTTIRegEx} "" editedFlags "${addedFlags}")
    string(REPLACE ${excptnRTTI_OFF} "" editedFlags "${editedFlags}")
    set_property(TARGET ${TARGET} PROPERTY COMPILE_FLAGS "${editedFlags}")

    get_property(addedFlags TARGET ${TARGET} PROPERTY COMPILE_DEFINITIONS)
    string(REGEX REPLACE ${excptnEHRTTIRegEx} "" editedFlags "${addedFlags}")
    string(REPLACE ${excptnRTTI_OFF} "" editedFlags "${editedFlags}")
    set_property(TARGET ${TARGET} PROPERTY COMPILE_DEFINITIONS "${editedFlags}")

    # Re-add the exception & rtti flags from LLVM
    set_property(SOURCE main.cpp APPEND_STRING PROPERTY COMPILE_FLAGS
                   " ${excptnExceptions_DFLT} ${excptnRTTI_DFLT} ")
    set_property(SOURCE Manager.cpp APPEND_STRING PROPERTY COMPILE_FLAGS
                   " ${excptnExceptions_DFLT} ${excptnRTTI_DFLT} ")

    # Invoke with exceptions & rtti
    set_property(SOURCE Invoke.cpp APPEND_STRING PROPERTY COMPILE_FLAGS
                   " ${excptnExceptions_ON} ${excptnRTTI_ON} ")

  endif()
endfunction(clang_enable_exceptions)

clang_enable_exceptions(clang-interpreter)
