foreach (scope DIRECTORY TARGET)
  define_property(${scope} PROPERTY LLDB_PLUGIN_KIND INHERITED
    BRIEF_DOCS "LLDB plugin kind (Process, SymbolFile, etc.)"
    FULL_DOCS  "See lldb/docs/resources/contributing.rst"
  )

  define_property(${scope} PROPERTY LLDB_ACCEPTABLE_PLUGIN_DEPENDENCIES INHERITED
    BRIEF_DOCS "LLDB plugin kinds which the plugin can depend on"
    FULL_DOCS  "See lldb/docs/resources/contributing.rst"
  )

  define_property(${scope} PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES INHERITED
    BRIEF_DOCS "LLDB plugin kinds which are depended on for historic reasons."
    FULL_DOCS  "See lldb/docs/resources/contributing.rst"
  )
endforeach()

option(LLDB_GENERATE_PLUGIN_DEP_GRAPH OFF)

function(check_lldb_plugin_layering)
  get_property(plugins GLOBAL PROPERTY LLDB_PLUGINS)
  foreach (plugin ${plugins})
    get_property(plugin_kind TARGET ${plugin} PROPERTY LLDB_PLUGIN_KIND)
    get_property(acceptable_deps TARGET ${plugin}
      PROPERTY LLDB_ACCEPTABLE_PLUGIN_DEPENDENCIES)
    get_property(tolerated_deps TARGET ${plugin}
      PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES)

    # A plugin is always permitted to depend on its own kind for the purposes
    # subclassing. Ideally the intra-kind dependencies should not form a loop,
    # but we're not checking that here.
    list(APPEND acceptable_deps ${plugin_kind})

    list(APPEND all_plugin_kinds ${plugin_kind})

    get_property(link_libs TARGET ${plugin} PROPERTY LINK_LIBRARIES)
    foreach (link_lib ${link_libs})
      if(link_lib IN_LIST plugins)
        get_property(lib_kind TARGET ${link_lib} PROPERTY LLDB_PLUGIN_KIND)
        if (lib_kind)
          if (lib_kind IN_LIST acceptable_deps)
            set(dep_kind green)
          elseif (lib_kind IN_LIST tolerated_deps)
            set(dep_kind yellow)
          else()
            set(dep_kind red)
            message(SEND_ERROR "Plugin ${plugin} cannot depend on ${lib_kind} "
              "plugin ${link_lib}")
          endif()
          list(APPEND dep_${dep_kind}_${plugin_kind}_${lib_kind} ${plugin})
        endif()
      endif()
    endforeach()
  endforeach()

  if (LLDB_GENERATE_PLUGIN_DEP_GRAPH)
    set(dep_graph "digraph Plugins {\n")
    list(REMOVE_DUPLICATES all_plugin_kinds)
    foreach (from ${all_plugin_kinds})
      foreach (to ${all_plugin_kinds})
        foreach (dep_kind green yellow red)
          if (dep_${dep_kind}_${from}_${to})
            list(REMOVE_DUPLICATES dep_${dep_kind}_${from}_${to})
            string(REGEX REPLACE "lldbPlugin|${from}" "" short_deps
              "${dep_${dep_kind}_${from}_${to}}")
            string(JOIN "\n" plugins ${short_deps})
            string(APPEND dep_graph
              "  ${from}->${to}[color=\"${dep_kind}\" label=\"${plugins}\"];\n")
          endif()
        endforeach()
      endforeach()
    endforeach()
    string(APPEND dep_graph "}\n")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/lldb-plugin-deps.dot" "${dep_graph}")
  endif()
endfunction()
