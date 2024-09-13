        .section .00cfg,"dr"
        .globl _load_config_used
        .p2align 3, 0
_load_config_used:
        .word 0x140
        .fill 0x54, 1, 0
        .xword __security_cookie
        .fill 0x10, 1, 0
        .xword __guard_check_icall_fptr
        .xword __guard_dispatch_icall_fptr
        .xword __guard_fids_table
        .xword __guard_fids_count
        .xword __guard_flags
        .xword 0
        .xword __guard_iat_table
        .xword __guard_iat_count
        .xword __guard_longjmp_table
        .xword __guard_longjmp_count
        .xword 0
        .xword __chpe_metadata
        .fill 0x78, 1, 0

__guard_check_icall_fptr:
        .xword 0
__guard_dispatch_icall_fptr:
        .xword 0
__os_arm64x_dispatch_call_no_redirect:
        .xword 0
__os_arm64x_dispatch_ret:
        .xword 0
__os_arm64x_check_call:
        .xword 0
        .globl __os_arm64x_dispatch_icall
__os_arm64x_dispatch_icall:
__os_arm64x_check_icall:
        .xword 0
__os_arm64x_get_x64_information:
        .xword 0
__os_arm64x_set_x64_information:
        .xword 0
__os_arm64x_check_icall_cfg:
        .xword 0
__os_arm64x_dispatch_fptr:
        .xword 0
__os_arm64x_helper0:
        .xword 0
__os_arm64x_helper1:
        .xword 0
__os_arm64x_helper2:
        .xword 0
__os_arm64x_helper3:
        .xword 0
__os_arm64x_helper4:
        .xword 0
__os_arm64x_helper5:
        .xword 0
__os_arm64x_helper6:
        .xword 0
__os_arm64x_helper7:
        .xword 0
__os_arm64x_helper8:
        .xword 0

        .data
        .globl __chpe_metadata
        .p2align 3, 0
__chpe_metadata:
        .word 1
        .rva __hybrid_code_map
        .word __hybrid_code_map_count
        .rva __x64_code_ranges_to_entry_points
        .rva __arm64x_redirection_metadata
        .rva __os_arm64x_dispatch_call_no_redirect
        .rva __os_arm64x_dispatch_ret
        .rva __os_arm64x_check_call
        .rva __os_arm64x_check_icall
        .rva __os_arm64x_check_icall_cfg
        .word 0 // __arm64x_native_entrypoint
        .rva __hybrid_auxiliary_iat
        .word __x64_code_ranges_to_entry_points_count
        .word __arm64x_redirection_metadata_count
        .rva __os_arm64x_get_x64_information
        .rva __os_arm64x_set_x64_information
        .rva __arm64x_extra_rfe_table
        .word __arm64x_extra_rfe_table_size
        .rva __os_arm64x_dispatch_fptr
        .word 0 // __hybrid_auxiliary_iat_copy
        .rva __os_arm64x_helper0
        .rva __os_arm64x_helper1
        .rva __os_arm64x_helper2
        .rva __os_arm64x_helper3
        .rva __os_arm64x_helper4
        .rva __os_arm64x_helper5
        .rva __os_arm64x_helper6
        .rva __os_arm64x_helper7
        .rva __os_arm64x_helper8

__security_cookie:
        .xword 0
