add_library(FortranLower
  PFTBuilder.cpp
)

target_link_libraries(FortranLower
  LLVMSupport
)

install (TARGETS FortranLower
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
