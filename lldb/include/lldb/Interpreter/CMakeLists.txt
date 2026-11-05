lldb_tablegen(CommandOptionArgumentTableEnums.inc -gen-lldb-option-enum-defs
  SOURCE CommandOptionArgumentTableEnums.td
  TARGET LLDBCommandOptionArgumentTableEnumsGen)

lldb_tablegen(CommandOptionArgumentTableEnums.json -dump-json
  SOURCE CommandOptionArgumentTableEnums.td
  TARGET LLDBCommandOptionArgumentTableEnumsJsonGen)
