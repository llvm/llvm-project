set(LLVM_OPTIONAL_SOURCES
  MaskableOpInterface.cpp
  MaskingOpInterface.cpp
  )

function(add_aiir_interface_library name)
  add_aiir_library(AIIR${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/Vector/Interfaces

    DEPENDS
    AIIR${name}IncGen

    LINK_LIBS PUBLIC
    AIIRIR
    )
endfunction(add_aiir_interface_library)

add_aiir_interface_library(MaskableOpInterface)
add_aiir_interface_library(MaskingOpInterface)
