# BMG/Xe2 payload contract — hardware-validated on Arc Pro B60 via register
# dumps (inter/probe/gen_dump_variants.py + launcher dump mode).

## Thread payload (what the COMPUTE_WALKER delivers)

- `r0.0` — indirect data blob address, low 32 bits. Mask with `~0x3F`
  (64B-aligned). Used as the a32 offset with stateless `bti[255]` loads.
- `r0.1`, `r0.6`, and `r0.7` — thread group IDs X, Y, and Z (ud).
- `r0.4[7:0]` — thread slot id within the dispatch.
- The software-local-ID entry receives the first 32 cross-thread bytes in
  `r1`, copies them after the local-ID GRFs, and loads local IDs into `r1-r3`.
  X-only, X/Y, and X/Y/Z layouts place the copy in `r2`, `r3`, and `r4`.
- When NEO enables hardware local-ID generation, it starts the kernel at
  `offset_to_skip_per_thread_data_load`, which is the encoded byte offset of
  the common-body label: the local-ID GRFs already contain data and the same
  inline payload is delivered in the following register.
- The common body reads inline data from that following register. Argument at
  zeinfo payload offset N lives at byte N of the register; a 64-bit pointer at
  offset 24 is subregister 3 with a qword operand.
- EU sub-register numbering is in units of the operand data type.

## Indirect data blob (in memory, at `r0.0 & ~0x3F`)

Layout (NEO strips the inline-mirrored prefix of cross-thread data):

```
+0x00  cross-thread remainder (args beyond the 32B inline mirror)
+0x20 + thread_slot*0xC0  per-thread data: packed local IDs, SoA u16 arrays:
         X at +0x20 (64B for SIMD32), Y at +0x60, Z at +0xA0
```

IGC's prologue loads, for reference:
- L1 `load.ugm.d32x32t.a32.ca.cc.bti[255]` (exdesc 0xFF000000, desc
  0x6229E500) at blob+0x20 -> local ID X+Y, 2 GRFs.
- L2 same msg `d32x16t` (desc 0x6219D500) at blob+0xA0 -> local ID Z, 1 GRF.
- L3 same msg `d32x8t` (desc 0x6219C500) at blob+0x00 -> cross-thread
  remainder (e.g. trailing pointer args), 1 GRF.

## Global ID

`gid.x = r0.1 * enqueued_local_size.x + local_id_x[lane] + global_id_offset.x`
(IGC: mul/macl for group*size, then add3 with the per-lane uw read of the
loaded local-ID array and the inline-register global offset).

## Message forms observed (Xe2, LSC/UGM)

- A64 scattered dword store: `send.ugm (32|M0) null rAddr rData:2 0x0
  0x08000584` (store.ugm.d32.a64). Address payload: 4 GRFs of per-lane qwords.
- A64 scattered dword load: `send.ugm (32|M0) rDst rAddr null:0 0x0
  0x08200580` (load.ugm.d32.a64), 2 GRF dst for 32 dwords.
- A64 single-dword load: desc 0x02108580 (load.ugm.d32x1t.a64).
- SIMD32 A64 atomic iadd: desc 0x0820058C (four address GRFs, two data GRFs,
  two result GRFs).
- Barrier signal payload: dword 2 is 0x100; bytes 10-11 copy bytes 11-12 of
  the inline-data register.
- EOT: `send.gtwy (1|M0) null r127 null:0 0x0 0x02000010 {EOT}`.

## SWSB notation seen in IGA text

`{@N}` any-pipe dist dep, `{I@N}` int-pipe, `{A@N}` ... , `{$t}` set token,
`{$t.src}`/`{$t.dst}` wait token as reader/writer, `{Compacted}`, `{EOT}`,
`sync.allrd null` = wait all pending reads.
