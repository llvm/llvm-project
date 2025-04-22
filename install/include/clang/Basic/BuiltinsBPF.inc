/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* List of builtins that Clang recognizes                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#ifdef GET_BUILTIN_ENUMERATORS
    BI__builtin_btf_type_id,
    BI__builtin_preserve_enum_value,
    BI__builtin_preserve_field_info,
    BI__builtin_preserve_type_info,

#endif // GET_BUILTIN_ENUMERATORS

#ifdef GET_BUILTIN_STR_TABLE

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif
static constexpr char BuiltinStringsStorage[] = 
  "\0"
  "__builtin_btf_type_id\0"
  "LUi.\0"
  "t\0"
  "__builtin_preserve_enum_value\0"
  "Li.\0"
  "__builtin_preserve_field_info\0"
  "Ui.\0"
  "__builtin_preserve_type_info\0"
  ;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

static constexpr llvm::StringTable BuiltinStrings =
    BuiltinStringsStorage;

#endif // GET_BUILTIN_STR_TABLE

#ifdef GET_BUILTIN_INFOS
    Builtin::Info{Builtin::Info::StrOffsets{1 /* __builtin_btf_type_id */, 23 /* LUi. */, 28 /* t */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{30 /* __builtin_preserve_enum_value */, 60 /* Li. */, 28 /* t */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{64 /* __builtin_preserve_field_info */, 94 /* Ui. */, 28 /* t */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
    Builtin::Info{Builtin::Info::StrOffsets{98 /* __builtin_preserve_type_info */, 23 /* LUi. */, 28 /* t */, 0 /*  */}, HeaderDesc::NO_HEADER, ALL_LANGUAGES},

#endif // GET_BUILTIN_INFOS

#ifdef GET_BUILTIN_PREFIXED_INFOS

#endif // GET_BUILTIN_PREFIXED_INFOS

#ifdef ATOMIC_BUILTIN

#endif // ATOMIC_BUILTIN
#undef ATOMIC_BUILTIN
