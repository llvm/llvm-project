include "EnumsBase.td"

def auto_download_enum_values : EnumDef<"lldb::eSymbolDownload", [
  EnumValue<"Off", "off", "Disable automatically downloading symbols.">,
  EnumValue<"Background", "background",
    "Download symbols in the background for images as they appear in the "
    "backtrace.">,
  EnumValue<"Foreground", "foreground",
    "Download symbols in the foreground for images as they appear in the "
    "backtrace.">,
]>;

def shared_cache_use_enum_values : EnumDef<"lldb::eSymbolSharedCacheUse", [
  EnumValue<"HostLLDBMemory", "host-lldb-memory",
            "Get binaries from the host lldb in-memory shared cache.">,
  EnumValue<"HostSharedCache", "host-shared-cache",
            "Get binaries from the host shared cache.">,
  EnumValue<"HostAndInferiorSharedCache", "host-and-inferior-shared-cache",
            "Get binaries from the host and inferior's shared caches.">,
  EnumValue<"InferiorSharedCacheOnly", "inferior-shared-cache-only",
            "Get binaries from inferior's shared cache only.">,
]>;
