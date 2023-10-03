#--- input_1.s
## Generated with:
##
##  - clang <input_file.c> -fsanitize=memtag-globals -O2 -S -fPIC -o - \
##          --target=aarch64-linux-android31 -fno-asynchronous-unwind-tables
##
## <input_file.c> contents:
##
##    /// Global variables defined here, of various semantics.
##    char global[30] = {};
##    __attribute__((no_sanitize("memtag"))) int global_untagged = 0;
##    const int const_global = 0;
##    static const int hidden_const_global = 0;
##    static char hidden_global[12] = {};
##    __attribute__((visibility("hidden"))) int hidden_attr_global = 0;
##    __attribute__((visibility("hidden"))) const int hidden_attr_const_global = 0;
##
##    /// Should be untagged.
##    __thread int tls_global;
##    __thread static int hidden_tls_global;
##
##    /// Tagged, from the other file.
##    extern int global_extern;
##    /// Untagged, from the other file.
##    extern __attribute__((no_sanitize("memtag"))) int global_extern_untagged;
##    /// Tagged, but from a different DSO (i.e. not this or the sister objfile).
##    extern int global_extern_outside_this_dso;
##    /// Tagged here (because it's non-const), but untagged in the definition found
##    /// in the sister objfile as it's marked as const there.
##    extern int global_extern_const_definition_but_nonconst_import;
##    /// Tagged here, but untagged in the definition found in the sister objfile
##    /// (explicitly).
##    extern int global_extern_untagged_definition_but_tagged_import;
##
##    /// ABS64 relocations. Also, forces symtab entries for local and external
##    /// globals.
##    char *pointer_to_global = &global[0];
##    char *pointer_inside_global = &global[17];
##    char *pointer_to_global_end = &global[30];
##    char *pointer_past_global_end = &global[48];
##    int *pointer_to_global_untagged = &global_untagged;
##    const int *pointer_to_const_global = &const_global;
##    /// RELATIVE relocations.
##    const int *pointer_to_hidden_const_global = &hidden_const_global;
##    char *pointer_to_hidden_global = &hidden_global[0];
##    const int *pointer_to_hidden_attr_global = &hidden_attr_global;
##    const int *pointer_to_hidden_attr_const_global = &hidden_attr_const_global;
##    /// RELATIVE relocations with special AArch64 MemtagABI semantics, with the
##    /// offset ('12' or '16') encoded in the place.
##    char *pointer_to_hidden_global_end = &hidden_global[12];
##    char *pointer_past_hidden_global_end = &hidden_global[16];
##    /// ABS64 relocations.
##    int *pointer_to_global_extern = &global_extern;
##    int *pointer_to_global_extern_untagged = &global_extern_untagged;
##    int *pointer_to_global_extern_outside_this_dso = &global_extern_outside_this_dso;
##    int *pointer_to_global_extern_const_definition_but_nonconst_import =
##        &global_extern_const_definition_but_nonconst_import;
##    int *pointer_to_global_extern_untagged_definition_but_tagged_import =
##        &global_extern_untagged_definition_but_tagged_import;
##
##    int *get_address_to_tls_global() { return &tls_global; }
##    int *get_address_to_hidden_tls_global() { return &hidden_tls_global; }

	.text
	.file	"a.c"
	.globl	get_address_to_tls_global       // -- Begin function get_address_to_tls_global
	.p2align	2
	.type	get_address_to_tls_global,@function
get_address_to_tls_global:              // @get_address_to_tls_global
// %bb.0:                               // %entry
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	mov	x29, sp
	adrp	x0, :tlsdesc:tls_global
	ldr	x1, [x0, :tlsdesc_lo12:tls_global]
	add	x0, x0, :tlsdesc_lo12:tls_global
	.tlsdesccall tls_global
	blr	x1
	mrs	x8, TPIDR_EL0
	add	x0, x8, x0
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	ret
.Lfunc_end0:
	.size	get_address_to_tls_global, .Lfunc_end0-get_address_to_tls_global
                                        // -- End function
	.globl	get_address_to_hidden_tls_global // -- Begin function get_address_to_hidden_tls_global
	.p2align	2
	.type	get_address_to_hidden_tls_global,@function
get_address_to_hidden_tls_global:       // @get_address_to_hidden_tls_global
// %bb.0:                               // %entry
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	mov	x29, sp
	adrp	x0, :tlsdesc:hidden_tls_global
	ldr	x1, [x0, :tlsdesc_lo12:hidden_tls_global]
	add	x0, x0, :tlsdesc_lo12:hidden_tls_global
	.tlsdesccall hidden_tls_global
	blr	x1
	mrs	x8, TPIDR_EL0
	add	x0, x8, x0
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	ret
.Lfunc_end1:
	.size	get_address_to_hidden_tls_global, .Lfunc_end1-get_address_to_hidden_tls_global
                                        // -- End function
	.memtag	global                          // @global
	.type	global,@object
	.bss
	.globl	global
	.p2align	4, 0x0
global:
	.zero	32
	.size	global, 32

	.type	global_untagged,@object         // @global_untagged
	.globl	global_untagged
	.p2align	2, 0x0
global_untagged:
	.word	0                               // 0x0
	.size	global_untagged, 4

	.type	const_global,@object            // @const_global
	.section	.rodata,"a",@progbits
	.globl	const_global
	.p2align	2, 0x0
const_global:
	.word	0                               // 0x0
	.size	const_global, 4

	.hidden	hidden_attr_global              // @hidden_attr_global
	.memtag	hidden_attr_global
	.type	hidden_attr_global,@object
	.bss
	.globl	hidden_attr_global
	.p2align	4, 0x0
hidden_attr_global:
	.zero	16
	.size	hidden_attr_global, 16

	.hidden	hidden_attr_const_global        // @hidden_attr_const_global
	.type	hidden_attr_const_global,@object
	.section	.rodata,"a",@progbits
	.globl	hidden_attr_const_global
	.p2align	2, 0x0
hidden_attr_const_global:
	.word	0                               // 0x0
	.size	hidden_attr_const_global, 4

	.memtag	pointer_to_global               // @pointer_to_global
	.type	pointer_to_global,@object
	.data
	.globl	pointer_to_global
	.p2align	4, 0x0
pointer_to_global:
	.xword	global
	.zero	8
	.size	pointer_to_global, 16

	.memtag	pointer_inside_global           // @pointer_inside_global
	.type	pointer_inside_global,@object
	.globl	pointer_inside_global
	.p2align	4, 0x0
pointer_inside_global:
	.xword	global+17
	.zero	8
	.size	pointer_inside_global, 16

	.memtag	pointer_to_global_end           // @pointer_to_global_end
	.type	pointer_to_global_end,@object
	.globl	pointer_to_global_end
	.p2align	4, 0x0
pointer_to_global_end:
	.xword	global+30
	.zero	8
	.size	pointer_to_global_end, 16

	.memtag	pointer_past_global_end         // @pointer_past_global_end
	.type	pointer_past_global_end,@object
	.globl	pointer_past_global_end
	.p2align	4, 0x0
pointer_past_global_end:
	.xword	global+48
	.zero	8
	.size	pointer_past_global_end, 16

	.memtag	pointer_to_global_untagged      // @pointer_to_global_untagged
	.type	pointer_to_global_untagged,@object
	.globl	pointer_to_global_untagged
	.p2align	4, 0x0
pointer_to_global_untagged:
	.xword	global_untagged
	.zero	8
	.size	pointer_to_global_untagged, 16

	.memtag	pointer_to_const_global         // @pointer_to_const_global
	.type	pointer_to_const_global,@object
	.globl	pointer_to_const_global
	.p2align	4, 0x0
pointer_to_const_global:
	.xword	const_global
	.zero	8
	.size	pointer_to_const_global, 16

	.type	hidden_const_global,@object     // @hidden_const_global
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
hidden_const_global:
	.word	0                               // 0x0
	.size	hidden_const_global, 4

	.memtag	pointer_to_hidden_const_global  // @pointer_to_hidden_const_global
	.type	pointer_to_hidden_const_global,@object
	.data
	.globl	pointer_to_hidden_const_global
	.p2align	4, 0x0
pointer_to_hidden_const_global:
	.xword	hidden_const_global
	.zero	8
	.size	pointer_to_hidden_const_global, 16

	.memtag	hidden_global                   // @hidden_global
	.type	hidden_global,@object
	.local	hidden_global
	.comm	hidden_global,16,16
	.memtag	pointer_to_hidden_global        // @pointer_to_hidden_global
	.type	pointer_to_hidden_global,@object
	.globl	pointer_to_hidden_global
	.p2align	4, 0x0
pointer_to_hidden_global:
	.xword	hidden_global
	.zero	8
	.size	pointer_to_hidden_global, 16

	.memtag	pointer_to_hidden_attr_global   // @pointer_to_hidden_attr_global
	.type	pointer_to_hidden_attr_global,@object
	.globl	pointer_to_hidden_attr_global
	.p2align	4, 0x0
pointer_to_hidden_attr_global:
	.xword	hidden_attr_global
	.zero	8
	.size	pointer_to_hidden_attr_global, 16

	.memtag	pointer_to_hidden_attr_const_global // @pointer_to_hidden_attr_const_global
	.type	pointer_to_hidden_attr_const_global,@object
	.globl	pointer_to_hidden_attr_const_global
	.p2align	4, 0x0
pointer_to_hidden_attr_const_global:
	.xword	hidden_attr_const_global
	.zero	8
	.size	pointer_to_hidden_attr_const_global, 16

	.memtag	pointer_to_hidden_global_end    // @pointer_to_hidden_global_end
	.type	pointer_to_hidden_global_end,@object
	.globl	pointer_to_hidden_global_end
	.p2align	4, 0x0
pointer_to_hidden_global_end:
	.xword	hidden_global+12
	.zero	8
	.size	pointer_to_hidden_global_end, 16

	.memtag	pointer_past_hidden_global_end  // @pointer_past_hidden_global_end
	.type	pointer_past_hidden_global_end,@object
	.globl	pointer_past_hidden_global_end
	.p2align	4, 0x0
pointer_past_hidden_global_end:
	.xword	hidden_global+16
	.zero	8
	.size	pointer_past_hidden_global_end, 16

	.memtag	global_extern
	.memtag	pointer_to_global_extern        // @pointer_to_global_extern
	.type	pointer_to_global_extern,@object
	.globl	pointer_to_global_extern
	.p2align	4, 0x0
pointer_to_global_extern:
	.xword	global_extern
	.zero	8
	.size	pointer_to_global_extern, 16

	.memtag	pointer_to_global_extern_untagged // @pointer_to_global_extern_untagged
	.type	pointer_to_global_extern_untagged,@object
	.globl	pointer_to_global_extern_untagged
	.p2align	4, 0x0
pointer_to_global_extern_untagged:
	.xword	global_extern_untagged
	.zero	8
	.size	pointer_to_global_extern_untagged, 16

	.memtag	global_extern_outside_this_dso
	.memtag	pointer_to_global_extern_outside_this_dso // @pointer_to_global_extern_outside_this_dso
	.type	pointer_to_global_extern_outside_this_dso,@object
	.globl	pointer_to_global_extern_outside_this_dso
	.p2align	4, 0x0
pointer_to_global_extern_outside_this_dso:
	.xword	global_extern_outside_this_dso
	.zero	8
	.size	pointer_to_global_extern_outside_this_dso, 16

	.memtag	global_extern_const_definition_but_nonconst_import
	.memtag	pointer_to_global_extern_const_definition_but_nonconst_import // @pointer_to_global_extern_const_definition_but_nonconst_import
	.type	pointer_to_global_extern_const_definition_but_nonconst_import,@object
	.globl	pointer_to_global_extern_const_definition_but_nonconst_import
	.p2align	4, 0x0
pointer_to_global_extern_const_definition_but_nonconst_import:
	.xword	global_extern_const_definition_but_nonconst_import
	.zero	8
	.size	pointer_to_global_extern_const_definition_but_nonconst_import, 16

	.memtag	global_extern_untagged_definition_but_tagged_import
	.memtag	pointer_to_global_extern_untagged_definition_but_tagged_import // @pointer_to_global_extern_untagged_definition_but_tagged_import
	.type	pointer_to_global_extern_untagged_definition_but_tagged_import,@object
	.globl	pointer_to_global_extern_untagged_definition_but_tagged_import
	.p2align	4, 0x0
pointer_to_global_extern_untagged_definition_but_tagged_import:
	.xword	global_extern_untagged_definition_but_tagged_import
	.zero	8
	.size	pointer_to_global_extern_untagged_definition_but_tagged_import, 16

	.type	tls_global,@object              // @tls_global
	.section	.tbss,"awT",@nobits
	.globl	tls_global
	.p2align	2, 0x0
tls_global:
	.word	0                               // 0x0
	.size	tls_global, 4

	.type	hidden_tls_global,@object       // @hidden_tls_global
	.p2align	2, 0x0
hidden_tls_global:
	.word	0                               // 0x0
	.size	hidden_tls_global, 4

	.ident	"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 6130c9df99a7a7eb9c6adc118a48f8f2acc534ab)"
	.section	".note.GNU-stack","",@progbits

#--- input_2.s
## Generated with:
##
##  - clang <input_file.c> -fsanitize=memtag-globals -O2 -S -o - \
##          --target=aarch64-linux-android31 -fno-asynchronous-unwind-tables
##
## <input_file.c> contents:
##
##     int global_extern;
##     static int global_extern_hidden;
##     __attribute__((no_sanitize("memtag"))) int global_extern_untagged;
##     const int global_extern_const_definition_but_nonconst_import;
##     __attribute__((no_sanitize(
##         "memtag"))) int global_extern_untagged_definition_but_tagged_import;
##

	.text
	.file	"b.c"
	.memtag	global_extern
	.type	global_extern,@object
	.bss
	.globl	global_extern
	.p2align	4, 0x0
global_extern:
	.zero	16
	.size	global_extern, 16

	.type	global_extern_untagged,@object
	.globl	global_extern_untagged
	.p2align	2, 0x0
global_extern_untagged:
	.word	0
	.size	global_extern_untagged, 4

	.type	global_extern_const_definition_but_nonconst_import,@object
	.section	.rodata,"a",@progbits
	.globl	global_extern_const_definition_but_nonconst_import
	.p2align	2, 0x0
global_extern_const_definition_but_nonconst_import:
	.word	0
	.size	global_extern_const_definition_but_nonconst_import, 4

	.type	global_extern_untagged_definition_but_tagged_import,@object
	.bss
	.globl	global_extern_untagged_definition_but_tagged_import
	.p2align	2, 0x0
global_extern_untagged_definition_but_tagged_import:
	.word	0
	.size	global_extern_untagged_definition_but_tagged_import, 4
