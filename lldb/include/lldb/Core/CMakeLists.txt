lldb_tablegen(Properties.inc -gen-lldb-property-defs
  SOURCE Properties.td
  TARGET LLDBPropertiesGen)

lldb_tablegen(PropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE Properties.td
  TARGET LLDBPropertiesEnumGen)
