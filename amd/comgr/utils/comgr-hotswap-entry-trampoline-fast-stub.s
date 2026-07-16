// Source for the B0->B0 entry-trampoline fast-path stub body. Assembled by
// gen-hotswap-fast-stub-inc.sh into comgr-hotswap-entry-trampoline-fast-stub.inc.
//
// The body is 40 bytes: the global_wb; v_nop unclaused-VMEM workaround prefix,
// then an s_get_pc / s_add64 / s_set_pc PC-relative redirect. The scratch pair
// is spelled s[100:101]; the runtime rewrites the six SGPR register-field bytes
// per kernel to the allocated pair (see comgr-hotswap-internal.h FastEntry*).
//
// The two s_add immediates use a non-zero literal so the assembler emits the
// 32-bit-literal encoding (imm=0 would use the shorter inline-constant form);
// the generator zeroes those dword slots, and the runtime writes the delta.
global_wb
v_nop
s_get_pc_i64 s[100:101]
s_add_co_u32 s100, s100, 0xdeadbeef
s_add_co_ci_u32 s101, s101, 0xdeadbeef
s_set_pc_i64 s[100:101]
