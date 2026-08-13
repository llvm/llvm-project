// RUN: inter-translate %S/../Integration/vadd.ll --import-llvm -o %t.mlir
// RUN: inter-opt %t.mlir '--inter-import-llvm=simd-width=32' -o %t.mlir
// RUN: inter-opt %t.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
// RUN: inter-translate %t.xemachine.mlir --xemachine-to-zebin -o %t.bin
// RUN: llvm-readobj --file-headers --sections --symbols %t.bin | FileCheck %s --check-prefix=ELF
// RUN: llvm-readobj --notes %t.bin | FileCheck %s --check-prefix=NOTE
// RUN: llvm-objcopy --dump-section=.ze_info=%t.yaml %t.bin
// RUN: FileCheck %s --check-prefix=ZE < %t.yaml
// RUN: inter-translate %S/../Integration/slm.ll --import-llvm -o %t.slm.mlir
// RUN: inter-opt %t.slm.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.slm.xemachine.mlir
// RUN: inter-translate %t.slm.xemachine.mlir --xemachine-to-zebin -o %t.slm.bin
// RUN: llvm-objcopy --dump-section=.ze_info=%t.slm.yaml %t.slm.bin
// RUN: FileCheck %s --check-prefix=SLM < %t.slm.yaml
// RUN: inter-translate %S/../Integration/atomic.ll --import-llvm -o %t.atomic.mlir
// RUN: inter-opt %t.atomic.mlir --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.atomic.xemachine.mlir
// RUN: inter-translate %t.atomic.xemachine.mlir --xemachine-to-zebin -o %t.atomic.bin
// RUN: llvm-objcopy --dump-section=.ze_info=%t.atomic.yaml %t.atomic.bin
// RUN: FileCheck %s --check-prefix=ATOMIC < %t.atomic.yaml

// ELF: Type: Processor Specific (0xff12)
// ELF: Machine: EM_INTELGT (0xCD)
// ELF: ProgramHeaderCount: 0
// ELF: SectionHeaderCount: 7
// ELF: Name: .text.vadd
// ELF: SHF_ALLOC
// ELF: SHF_EXECINSTR
// ELF: Name: .symtab
// ELF: Name: .strtab
// ELF: Name: .ze_info
// ELF: Type: Unknown (0xFF000011)
// ELF: Name: .note.intelgt.compat
// ELF: Name: .shstrtab
// ELF: Name: vadd
// ELF: Binding: Global
// ELF: Type: Function
// ELF: Section: .text.vadd

// NOTE: Owner: IntelGT
// NOTE: FA040000
// NOTE: Owner: IntelGT
// NOTE: 090C0000
// NOTE: Owner: IntelGT
// NOTE: 00000000
// NOTE: Owner: IntelGT
// NOTE: 312E3634 00
// NOTE: Owner: IntelGT
// NOTE: 00400005

// ZE: version: '1.64'
// ZE: name: 'vadd'
// ZE: grf_count: 128
// ZE: has_4gb_buffers: true
// ZE: has_no_stateless_write: false
// ZE: inline_data_payload_size: 32
// ZE: offset_to_skip_per_thread_data_load: 144
// ZE: simd_size: 32
// ZE: arg_type: global_id_offset
// ZE: arg_type: enqueued_local_size
// ZE: arg_type: arg_bypointer
// ZE-NEXT: offset: 24
// ZE-NEXT: size: 8
// ZE-NEXT: arg_index: 0
// ZE: arg_type: arg_bypointer
// ZE-NEXT: offset: 32
// ZE-NEXT: size: 8
// ZE-NEXT: arg_index: 1
// ZE: arg_type: arg_bypointer
// ZE-NEXT: offset: 40
// ZE-NEXT: size: 8
// ZE-NEXT: arg_index: 2
// ZE: per_thread_payload_arguments:
// ZE: arg_type: local_id
// ZE: size: 64

// SLM: name: 'slm_kernel'
// SLM: barrier_count: 1
// SLM: slm_size: 128

// ATOMIC: name: 'atomic_kernel'
// ATOMIC: has_global_atomics: true
// ATOMIC: has_no_stateless_write: false
