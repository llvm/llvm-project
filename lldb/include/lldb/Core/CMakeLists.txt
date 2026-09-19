lldb_tablegen(ModuleListEnums.inc -gen-lldb-option-enum-defs
  SOURCE ModuleListEnums.td
  TARGET LLDBModuleListEnumsGen)

lldb_tablegen(ModuleListEnums.json -dump-json
  SOURCE ModuleListEnums.td
  TARGET LLDBModuleListEnumsJsonGen)
