* `2.5`: Introduce `amd_comgr_populate_mangled_names` and
  `amd_comgr_get_mangled_name` APIS.
* `2.4`: Introduce `amd_comgr_create_symbolizer_info`, `amd_comgr_symbolize`,
  `amd_comgr_destroy_symbolizer_info` APIS.
* `2.3`: Introduce `amd_comgr_set_data_from_file_slice` and
  `amd_comgr_lookup_code_object` APIS.
* `2.2`: Introduce `amd_comgr_demangle_symbol_name` API.
* `2.1`: Add `AMD_COMGR_TIME_STATISTICS` environment variable.
* `2.0`: Add support for new target feature syntax introduced at [AMDGPUUsage](https://llvm.org/docs/AMDGPUUsage.html).
* `1.9`: Add gfx1031
* `1.8`: Implement GNU Symbol Versioning for all exported functions. Rename
  some macros exposed in `amd_comgr.h` to avoid conflicts.
* `1.7`: Add `AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC`, a
  replacement for `AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES`, which is now
  deprecated.
* `1.6`: Add `AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL` for Code Object V2
  kernel symbols.
* `1.5`: Add `AMD_COMGR_SYMBOL_TYPE_UNKNOWN` for unknown/unsupported ELF symbol
  types. This fixes a bug where these symbols were previously reported as
  `AMD_COMGR_SYMBOL_TYPE_NOTYPE`.
* `1.4`: Support out-of-process HIP compilation to fat binary.
* `1.3`: Introduce `amd_comgr_action_info_set_option_list`,
  `amd_comgr_action_info_get_option_list_count`, and
  `amd_comgr_action_info_get_option_list_item` to replace the old option APIs
  `amd_comgr_action_info_set_options` and `amd_comgr_action_info_get_options`.
  The old APIs do not support arguments with embedded delimiters, and are
  replaced with an array-oriented API. The old APIs are deprecated and will be
  removed in a future version of the library.
* `1.2`: Introduce `amd_comgr_disassemble_instruction` and associated APIS.
* `1.1`: First versioned release. Versions before this have no guaranteed
  compatibility.
