option(CLANG_INSTALL_SCANVIEW "Install the scan-view tool" ON)

set(BinFiles
      Reporter.py
      ScanView.py
      scan-view
      startfile.py)

set(ResourceFiles
      FileRadar.scpt
      GetRadarVersion.scpt
      bugcatcher.ico)

if(CLANG_INSTALL_SCANVIEW)
  foreach(BinFile ${BinFiles})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/bin/${BinFile}
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/bin
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${BinFile}
                         ${CMAKE_BINARY_DIR}/bin/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${BinFile})
    list(APPEND Depends ${CMAKE_BINARY_DIR}/bin/${BinFile})
    install(PROGRAMS ${BinFile} DESTINATION bin)
  endforeach()

  foreach(ResourceFile ${ResourceFiles})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/share/scan-view/${ResourceFile}
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/share/scan-view
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/Resources/${ResourceFile}
                         ${CMAKE_BINARY_DIR}/share/scan-view/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Resources/${ResourceFile})
    list(APPEND Depends ${CMAKE_BINARY_DIR}/share/scan-view/${ResourceFile})
    install(FILES Resources/${ResourceFile} DESTINATION share/scan-view)
  endforeach()

  add_custom_target(scan-view ALL DEPENDS ${Depends})
  set_target_properties(scan-view PROPERTIES FOLDER "Misc")
endif()
