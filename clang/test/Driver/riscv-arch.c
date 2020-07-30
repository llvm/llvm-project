// RUN: %clang -target riscv32-unknown-elf -march=rv32i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32im -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32ima -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32ic -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32imafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32ia -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32iac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32iafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv32-unknown-elf -march=rv32g -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv32-unknown-elf -march=rv32gc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64im -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64ima -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64ic -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64imafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64ia -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64iac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64iafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang -target riscv64-unknown-elf -march=rv64g -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64gc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// CHECK-NOT: error: invalid arch name '

// RUN: %clang -target riscv32-unknown-elf -march=rv32 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32 %s
// RV32: error: invalid arch name 'rv32'

// RUN: %clang -target riscv32-unknown-elf -march=rv32m -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32M %s
// RV32M: error: invalid arch name 'rv32m'

// RUN: %clang -target riscv32-unknown-elf -march=rv32id -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32ID %s
// RV32ID: error: invalid arch name 'rv32id'

// RUN: %clang -target riscv32-unknown-elf -march=rv32l -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32L %s
// RV32L: error: invalid arch name 'rv32l'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imadf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32IMADF %s
// RV32IMADF: error: invalid arch name 'rv32imadf'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imm -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32IMM %s
// RV32IMM: error: invalid arch name 'rv32imm'

// RUN: %clang -target riscv32-unknown-elf -march=RV32I -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32I-UPPER %s
// RV32I-UPPER: error: invalid arch name 'RV32I'

// RUN: %clang -target riscv64-unknown-elf -march=rv64 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64 %s
// RV64: error: invalid arch name 'rv64'

// RUN: %clang -target riscv64-unknown-elf -march=rv64m -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64M %s
// RV64M: error: invalid arch name 'rv64m'

// RUN: %clang -target riscv64-unknown-elf -march=rv64id -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64ID %s
// RV64ID: error: invalid arch name 'rv64id'

// RUN: %clang -target riscv64-unknown-elf -march=rv64l -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64L %s
// RV64L: error: invalid arch name 'rv64l'

// RUN: %clang -target riscv64-unknown-elf -march=rv64imadf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64IMADF %s
// RV64IMADF: error: invalid arch name 'rv64imadf'

// RUN: %clang -target riscv64-unknown-elf -march=rv64imm -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64IMM %s
// RV64IMM: error: invalid arch name 'rv64imm'

// RUN: %clang -target riscv64-unknown-elf -march=RV64I -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64I-UPPER %s
// RV64I-UPPER: error: invalid arch name 'RV64I'


// Testing specific messages and unsupported extensions.

// RUN: %clang -target riscv32-unknown-elf -march=rv32e -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32E %s
// RV32E: error: invalid arch name 'rv32e',
// RV32E: standard user-level extension 'e'

// RUN: %clang -target riscv64-unknown-elf -march=rv64e -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64E %s
// RV64E: error: invalid arch name 'rv64e',
// RV64E: standard user-level extension 'e' requires 'rv32'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imC -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-LOWER %s
// RV32-LOWER: error: invalid arch name 'rv32imC',
// RV32-LOWER: string must be lowercase

// RUN: %clang -target riscv32-unknown-elf -march=rv32 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-STR %s
// RV32-STR: error: invalid arch name 'rv32',
// RV32-STR: string must begin with rv32{i,e,g} or rv64{i,g}

// RUN: %clang -target riscv32-unknown-elf -march=rv32q -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-LETTER %s
// RV32-LETTER: error: invalid arch name 'rv32q',
// RV32-LETTER: first letter should be 'e', 'i' or 'g'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imcq -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ORDER %s
// RV32-ORDER: error: invalid arch name 'rv32imcq',
// RV32-ORDER: standard user-level extension not given in canonical order 'q'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imw -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-STD-INVAL %s
// RV32-STD-INVAL: error: invalid arch name 'rv32imw',
// RV32-STD-INVAL: invalid standard user-level extension 'w'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imqc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-STD %s
// RV32-STD: error: invalid arch name 'rv32imqc',
// RV32-STD: unsupported standard user-level extension 'q'

// RUN: %clang -target riscv32-unknown-elf -march=rv32xabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32X %s
// RV32X: error: invalid arch name 'rv32xabc',
// RV32X: first letter should be 'e', 'i' or 'g'

// RUN: %clang -target riscv32-unknown-elf -march=rv32sxabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32SX %s
// RV32SX: error: invalid arch name 'rv32sxabc',
// RV32SX: first letter should be 'e', 'i' or 'g'

// RUN: %clang -target riscv32-unknown-elf -march=rv32sabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32S %s
// RV32S: error: invalid arch name 'rv32sabc',
// RV32S: first letter should be 'e', 'i' or 'g'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ix -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32X-NAME %s
// RV32X-NAME: error: invalid arch name 'rv32ix',
// RV32X-NAME: non-standard user-level extension name missing after 'x'

// RUN: %clang -target riscv32-unknown-elf -march=rv32isx -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32SX-NAME %s
// RV32SX-NAME: error: invalid arch name 'rv32isx',
// RV32SX-NAME: non-standard supervisor-level extension
// RV32SX-NAME: name missing after 'sx'

// RUN: %clang -target riscv32-unknown-elf -march=rv32is -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32S-NAME %s
// RV32S-NAME: error: invalid arch name 'rv32is',
// RV32S-NAME: standard supervisor-level extension
// RV32S-NAME: name missing after 's'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ix_s_sx -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32ALL-NAME %s
// RV32ALL-NAME: error: invalid arch name 'rv32ix_s_sx',
// RV32ALL-NAME: non-standard user-level extension
// RV32ALL-NAME: name missing after 'x'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32X-UNS %s
// RV32X-UNS: error: invalid arch name 'rv32ixabc',
// RV32X-UNS: unsupported non-standard user-level extension 'xabc'

// RUN: %clang -target riscv32-unknown-elf -march=rv32isa -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32S-UNS %s
// RV32S-UNS: error: invalid arch name 'rv32isa',
// RV32S-UNS: unsupported standard supervisor-level extension 'sa'

// RUN: %clang -target riscv32-unknown-elf -march=rv32isxabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32SX-UNS %s
// RV32SX-UNS: error: invalid arch name 'rv32isxabc',
// RV32SX-UNS: unsupported non-standard supervisor-level extension 'sxabc'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixabc_sp_sxlw -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32ALL %s
// RV32ALL: error: invalid arch name 'rv32ixabc_sp_sxlw',
// RV32ALL: unsupported non-standard user-level extension 'xabc'

// RUN: %clang -target riscv32-unknown-elf -march=rv32i20 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-IVER %s
// RV32-IVER: error: invalid arch name 'rv32i20', unsupported
// RV32-IVER: version number 20 for extension 'i'

// RUN: %clang -target riscv32-unknown-elf -march=rv32imc5 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-CVER %s
// RV32-CVER: error: invalid arch name 'rv32imc5', unsupported
// RV32-CVER: version number 5 for extension 'c'

// RUN: %clang -target riscv32-unknown-elf -march=rv32i2p -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-IMINOR-MISS %s
// RV32-IMINOR-MISS: error: invalid arch name 'rv32i2p',
// RV32-IMINOR-MISS: minor version number missing after 'p' for extension 'i'

// RUN: %clang -target riscv32-unknown-elf -march=rv32i2p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-IMINOR0 %s
// RV32-IMINOR0: error: invalid arch name 'rv32i2p0',
// RV32-IMINOR0: unsupported version number 2.0 for extension 'i'

// RUN: %clang -target riscv32-unknown-elf -march=rv32i2p1 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-IMINOR1 %s
// RV32-IMINOR1: error: invalid arch name 'rv32i2p1', unsupported
// RV32-IMINOR1: version number 2.1 for extension 'i'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixt2p -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-XMINOR-MISS %s
// RV32-XMINOR-MISS: error: invalid arch name 'rv32ixt2p',
// RV32-XMINOR-MISS: minor version number missing after 'p' for extension 'xt'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ist2p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-SMINOR0 %s
// RV32-SMINOR0: error: invalid arch name 'rv32ist2p0',
// RV32-SMINOR0: unsupported version number 2.0 for extension 'st'

// RUN: %clang -target riscv32-unknown-elf -march=rv32isxt2p1 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-SXMINOR1 %s
// RV32-SXMINOR1: error: invalid arch name 'rv32isxt2p1', unsupported
// RV32-SXMINOR1: version number 2.1 for extension 'sxt'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixabc_ -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-XSEP %s
// RV32-XSEP: error: invalid arch name 'rv32ixabc_',
// RV32-XSEP: extension name missing after separator '_'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixabc_a -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-PREFIX %s
// RV32-PREFIX: error: invalid arch name 'rv32ixabc_a',
// RV32-PREFIX: invalid extension prefix 'a'

// RUN: %clang -target riscv32-unknown-elf -march=rv32isabc_xdef -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-X-ORDER %s
// RV32-X-ORDER: error: invalid arch name 'rv32isabc_xdef',
// RV32-X-ORDER: non-standard user-level extension not given
// RV32-X-ORDER: in canonical order 'xdef'

// RUN: %clang -target riscv32-unknown-elf -march=rv32isxabc_sdef -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-S-ORDER %s
// RV32-S-ORDER: error: invalid arch name 'rv32isxabc_sdef',
// RV32-S-ORDER: standard supervisor-level extension not given
// RV32-S-ORDER: in canonical order 'sdef'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixabc_xabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-XDUP %s
// RV32-XDUP: error: invalid arch name 'rv32ixabc_xabc',
// RV32-XDUP: duplicated non-standard user-level extension 'xabc'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixabc_xdef -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-X-X-INVAL %s
// RV32-X-X-INVAL: error: invalid arch name 'rv32ixabc_xdef', unsupported
// RV32-X-X-INVAL: non-standard user-level extension 'xabc'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ixabc_sdef_sxghi -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-X-S-SX-INVAL %s
// RV32-X-S-SX-INVAL: error: invalid arch name 'rv32ixabc_sdef_sxghi',
// RV32-X-S-SX-INVAL: unsupported non-standard user-level extension 'xabc'

// RUN: %clang -target riscv32-unknown-elf -march=rv32i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-TARGET %s
// RUN: %clang -target riscv64-unknown-elf -march=rv32i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-TARGET %s
// RV32-TARGET: "-triple" "riscv32-unknown-unknown-elf"

// RUN: %clang -target riscv32-unknown-elf -march=rv64i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64-TARGET %s
// RUN: %clang -target riscv64-unknown-elf -march=rv64i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64-TARGET %s
// RV64-TARGET: "-triple" "riscv64-unknown-unknown-elf"

// RUN: %clang -target riscv32-unknown-elf -march=rv32ib -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-NOFLAG %s
// RV32-EXPERIMENTAL-NOFLAG: error: invalid arch name 'rv32ib'
// RV32-EXPERIMENTAL-NOFLAG: requires '-menable-experimental-extensions'

// RUN: %clang -target riscv32-unknown-elf -march=rv32ib -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-NOVERS %s
// RV32-EXPERIMENTAL-NOVERS: error: invalid arch name 'rv32ib'
// RV32-EXPERIMENTAL-NOVERS: experimental extension requires explicit version number

// RUN: %clang -target riscv32-unknown-elf -march=rv32ib0p1 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-BADVERS %s
// RV32-EXPERIMENTAL-BADVERS: error: invalid arch name 'rv32ib0p1'
// RV32-EXPERIMENTAL-BADVERS: unsupported version number 0.1 for experimental extension

// RUN: %clang -target riscv32-unknown-elf -march=rv32ib0p92 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-GOODVERS %s
// RV32-EXPERIMENTAL-GOODVERS: "-target-feature" "+experimental-b"

// RUN: %clang -target riscv32-unknown-elf -march=rv32izbb -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-ZBB-NOFLAG %s
// RV32-EXPERIMENTAL-ZBB-NOFLAG: error: invalid arch name 'rv32izbb'
// RV32-EXPERIMENTAL-ZBB-NOFLAG: requires '-menable-experimental-extensions'

// RUN: %clang -target riscv32-unknown-elf -march=rv32izbb0p92 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-ZBB %s
// RV32-EXPERIMENTAL-ZBB: "-target-feature" "+experimental-zbb"

// RUN: %clang -target riscv32-unknown-elf -march=rv32izbb0p92_zbp0p92 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-ZBB-ZBP %s
// RV32-EXPERIMENTAL-ZBB-ZBP: "-target-feature" "+experimental-zbb"
// RV32-EXPERIMENTAL-ZBB-ZBP: "-target-feature" "+experimental-zbp"

// RUN: %clang -target riscv32-unknown-elf -march=rv32iv -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-EXPERIMENTAL-V-NOFLAG %s
// RV32-EXPERIMENTAL-V-NOFLAG: error: invalid arch name 'rv32iv'
// RV32-EXPERIMENTAL-V-NOFLAG: requires '-menable-experimental-extensions'

// RUN: %clang -target riscv32-unknown-elf -march=rv32iv -menable-experimental-extensions -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-EXPERIMENTAL-V-NOVERS %s
// RV32-EXPERIMENTAL-V-NOVERS: error: invalid arch name 'rv32iv'
// RV32-EXPERIMENTAL-V-NOVERS: experimental extension requires explicit version number

// RUN: %clang -target riscv32-unknown-elf -march=rv32iv0p1 -menable-experimental-extensions -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-EXPERIMENTAL-V-BADVERS %s
// RV32-EXPERIMENTAL-V-BADVERS: error: invalid arch name 'rv32iv0p1'
// RV32-EXPERIMENTAL-V-BADVERS: unsupported version number 0.1 for experimental extension

// RUN: %clang -target riscv32-unknown-elf -march=rv32iv0p8 -menable-experimental-extensions -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-EXPERIMENTAL-V-GOODVERS %s
// RV32-EXPERIMENTAL-V-GOODVERS: "-target-feature" "+experimental-v"
