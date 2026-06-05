/* merge.ld — Partial-link script for EJIT lipo .a
 *
 * Merges all .o files from the lipo .a into a single relocatable .o,
 * discarding .group (COMDAT) metadata and other link-time-only sections.
 *
 * Usage:
 *   ld.lld -r -o merged.o --script=merge.ld --whole-archive libejit_lipo.a
 */

SECTIONS
{
  /* ── DISCARD ──────────────────────────────────────────────────────── */
  /DISCARD/ : {
    *(.group .group.*)        /*  COMDAT dedup records              */
    *(.rela.debug*)           /*  debug relocations (not needed)     */
    *(.rela.llvm_addrsig*)    /*  address-significance relocs        */
    *(*.llvm_addrsig*)        /*  any leftover addrsig sections      */
  }

  /* ── code ─────────────────────────────────────────────────────────── */
  .text           : { *(.text .text.* .gnu.linkonce.t.*) }
  .rodata         : { *(.rodata .rodata.* .gnu.linkonce.r.*) }
  .data           : { *(.data .data.* .gnu.linkonce.d.*) }
  .bss            : { *(.bss .bss.* .gnu.linkonce.b.*) }

  /* ── init / fini ──────────────────────────────────────────────────── */
  .init           : { *(.init) }
  .fini           : { *(.fini) }
  .init_array     : { *(.init_array .init_array.*) }
  .fini_array     : { *(.fini_array .fini_array.*) }

  /* ── C++ / unwind ─────────────────────────────────────────────────── */
  .eh_frame       : { *(.eh_frame .eh_frame.*) }
  .gcc_except_table : { *(.gcc_except_table .gcc_except_table.*) }
  .tbss           : { *(.tbss .tbss.*) }
  .tdata          : { *(.tdata .tdata.*) }

  /* ── read-write data ──────────────────────────────────────────────── */
  .data.rel.ro    : { *(.data.rel.ro .data.rel.ro.*) }
  .dynamic        : { *(.dynamic) }
  .got            : { *(.got .got.*) }
  .got.plt        : { *(.got.plt) }
  .plt            : { *(.plt .plt.*) }

  /* ── misc known sections ──────────────────────────────────────────── */
  .interp         : { *(.interp) }
  .note           : { *(.note .note.*) }
  .comment        : { *(.comment .comment.*) }
  .llvm_addrsig   : { *(.llvm_addrsig) }
  .llvm.call_graph_profile : { *(.llvm.call_graph_profile) }
  .cg_profile      : { *(.cg_profile) }
  .ctors          : { *(.ctors .ctors.*) }
  .dtors          : { *(.dtors .dtors.*) }
  .tm_clone_table : { *(.tm_clone_table) }

  /* ── relocations (keep ― needed by partial link) ──────────────────── */
  .rela.text      : { *(.rela.text .rela.text.*) }
  .rela.rodata    : { *(.rela.rodata .rela.rodata.*) }
  .rela.data      : { *(.rela.data .rela.data.*) }
  .rela.data.rel.ro : { *(.rela.data.rel.ro .rela.data.rel.ro.*) }
  .rela.eh_frame  : { *(.rela.eh_frame .rela.eh_frame.*) }
  .rela.init_array : { *(.rela.init_array .rela.init_array.*) }
  .rela.plt       : { *(.rela.plt) }

}
