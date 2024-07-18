if(CMAKE_CROSSCOMPILING)
    set(LLVM_CROSSCOMPILING TRUE)
else()
    if(APPLE)
        foreach(arch ${CMAKE_OSX_ARCHITECTURES})
            execute_process(COMMAND arch -arch ${arch} true
                            RESULT_VARIABLE ARCH_SUPPORTED
                            OUTPUT_QUIET
                            ERROR_QUIET)

            if(ARCH_SUPPORTED EQUAL 0)
                set(CURRENT_ARCH_SUPPORTED TRUE)
                break()
            endif()
        endforeach()
        if(NOT CURRENT_ARCH_SUPPORTED)
            set(LLVM_CROSSCOMPILING TRUE)
        endif()
    endif()
endif()
