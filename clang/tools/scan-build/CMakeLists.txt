option(CLANG_INSTALL_SCANBUILD "Install the scan-build tool" ON)

if (WIN32 AND NOT CYGWIN)
  set(BinFiles
        scan-build.bat)
  set(LibexecFiles
        ccc-analyzer.bat
        c++-analyzer.bat)
else()
  set(BinFiles
        scan-build)
  set(LibexecFiles
        ccc-analyzer
        c++-analyzer)
  if (APPLE)
    list(APPEND BinFiles
           set-xcode-analyzer)
  endif()
endif()

set(ManPages
      scan-build.1)

set(ShareFiles
      scanview.css
      sorttable.js)


if(CLANG_INSTALL_SCANBUILD)
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

  foreach(LibexecFile ${LibexecFiles})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/libexec/${LibexecFile}
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/libexec
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${LibexecFile}
                         ${CMAKE_BINARY_DIR}/libexec/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${LibexecFile})
    list(APPEND Depends ${CMAKE_BINARY_DIR}/libexec/${LibexecFile})
    install(PROGRAMS ${LibexecFile} DESTINATION libexec)
  endforeach()

  foreach(ManPage ${ManPages})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/share/man/man1/${ManPage}
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/share/man/man1
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/${ManPage}
                         ${CMAKE_BINARY_DIR}/share/man/man1/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ManPage})
    list(APPEND Depends ${CMAKE_BINARY_DIR}/share/man/man1/${ManPage})
    install(PROGRAMS ${ManPage} DESTINATION share/man/man1)
  endforeach()

  foreach(ShareFile ${ShareFiles})
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/share/scan-view/${ShareFile}
                       COMMAND ${CMAKE_COMMAND} -E make_directory
                         ${CMAKE_BINARY_DIR}/share/scan-view
                       COMMAND ${CMAKE_COMMAND} -E copy
                         ${CMAKE_CURRENT_SOURCE_DIR}/share/${ShareFile}
                         ${CMAKE_BINARY_DIR}/share/scan-view/
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/share/${ShareFile})
    list(APPEND Depends ${CMAKE_BINARY_DIR}/share/scan-view/${ShareFile})
    install(FILES ${ShareFile} DESTINATION share/scan-view)
  endforeach()

  add_custom_target(scan-build ALL DEPENDS ${Depends})
  set_target_properties(scan-build PROPERTIES FOLDER "Misc")
endif()

