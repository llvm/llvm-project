add_aiir_dialect(X86 x86)
add_aiir_doc(X86 X86 Dialects/ -gen-dialect-doc -dialect=x86)

add_aiir_interface(X86Interfaces)
add_dependencies(AIIRX86IncGen AIIRX86InterfacesIncGen)

add_subdirectory(TransformOps)
