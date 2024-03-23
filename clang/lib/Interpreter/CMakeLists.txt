set(LLVM_LINK_COMPONENTS
   core
   native
   MC
   Option
   OrcJit
   OrcDebugging
   OrcShared
   OrcTargetProcess
   Support
   Target
   TargetParser
  )

add_clang_library(clangInterpreter
  DeviceOffload.cpp
  CodeCompletion.cpp
  IncrementalExecutor.cpp
  IncrementalParser.cpp
  Interpreter.cpp
  InterpreterUtils.cpp
  Value.cpp

  DEPENDS
  intrinsics_gen
  ClangDriverOptions

  LINK_LIBS
  clangAST
  clangAnalysis
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSema
  clangSerialization
  )

if ((MINGW OR CYGWIN) AND BUILD_SHARED_LIBS)
  # The DLLs are supposed to export all symbols (except for ones that are
  # explicitly hidden). Normally, this is what happens anyway, but if there
  # are symbols that are marked explicitly as dllexport, we'd only export them
  # and nothing else. The Interpreter contains a few cases of such dllexports
  # (for symbols that need to be exported even from standalone exe files);
  # therefore, add --export-all-symbols to make sure we export all symbols
  # despite potential dllexports.
  target_link_options(clangInterpreter PRIVATE LINKER:--export-all-symbols)
endif()

if(MSVC)
  set_target_properties(clangInterpreter PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)

  # RTTI/C++ symbols
  set(clangInterpreter_exports ${clangInterpreter_exports} ??_7type_info@@6B@
    ?__type_info_root_node@@3U__type_info_node@@A
    ?nothrow@std@@3Unothrow_t@1@B
  )

  # Compiler added symbols for static variables. NOT for VStudio < 2015
  set(clangInterpreter_exports ${clangInterpreter_exports} _Init_thread_abort _Init_thread_epoch
    _Init_thread_footer _Init_thread_header _tls_index
  )

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # new/delete variants needed when linking to static msvc runtime (esp. Debug)
    set(clangInterpreter_exports ${clangInterpreter_exports}
      ??2@YAPEAX_K@Z
      ??3@YAXPEAX@Z
      ??_U@YAPEAX_K@Z
      ??_V@YAXPEAX@Z
      ??3@YAXPEAX_K@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@PEBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@PEBD@Z
      ?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z
    )
  else()
    set(clangInterpreter_exports ${clangInterpreter_exports}
      ??2@YAPAXI@Z
      ??3@YAXPAX@Z
      ??3@YAXPAXI@Z
      ??_U@YAPAXI@Z
      ??_V@YAXPAX@Z
      ??_V@YAXPAXI@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@H@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@M@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@N@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@PBX@Z
      ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QAEAAV01@P6AAAV01@AAV01@@Z@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@D@Z
      ??$?6U?$char_traits@D@std@@@std@@YAAAV?$basic_ostream@DU?$char_traits@D@std@@@0@AAV10@PBD@Z
      ?_Facet_Register@std@@YAXPAV_Facet_base@1@@Z
    )
  endif()

  # List to '/EXPORT:sym0 /EXPORT:sym1 /EXPORT:sym2 ...'
  foreach(sym ${clangInterpreter_exports})
    set(clangInterpreter_link_str "${clangInterpreter_link_str} /EXPORT:${sym}")
  endforeach(sym ${clangInterpreter_exports})

  target_link_options(clangInterpreter PUBLIC "${clangInterpreter_link_str}")
endif(MSVC)
