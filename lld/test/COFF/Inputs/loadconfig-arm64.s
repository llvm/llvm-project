        .section .rdata,"dr"
        .globl _load_config_used
        .p2align 3, 0
_load_config_used:
        .word 0x140
        .fill 0x7c,1,0
        .xword __guard_fids_table
        .xword __guard_fids_count
        .xword __guard_flags
        .xword 0
        .xword __guard_iat_table
        .xword __guard_iat_count
        .xword __guard_longjmp_table
        .xword __guard_longjmp_count
        .fill 0x80,1,0
