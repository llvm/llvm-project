// RUN: %clang --target=riscv32-unknown-elf -march=rv32i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32i2p1 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32im -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32ima -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32imaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32imafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv32-unknown-elf -march=rv32ic -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32imc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32imac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32imafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32imafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv32-unknown-elf -march=rv32ia -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32iaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32iafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv32-unknown-elf -march=rv32iac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32iafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32iafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv32-unknown-elf -march=rv32g -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32gc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv32-unknown-elf -mabi=ilp32 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-ILP32 %s

// CHECK-ILP32:      "-target-feature" "+m"
// CHECK-ILP32-SAME: {{^}} "-target-feature" "+a"
// CHECK-ILP32-SAME: {{^}} "-target-feature" "+f"
// CHECK-ILP32-SAME: {{^}} "-target-feature" "+d"
// CHECK-ILP32-SAME: {{^}} "-target-feature" "+c"

// RUN: %clang --target=riscv32-unknown-elf -mabi=ilp32f -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-ILP32F %s

// CHECK-ILP32F:      "-target-feature" "+m"
// CHECK-ILP32F-SAME: {{^}} "-target-feature" "+a"
// CHECK-ILP32F-SAME: {{^}} "-target-feature" "+f"
// CHECK-ILP32F-SAME: {{^}} "-target-feature" "+d"
// CHECK-ILP32F-SAME: {{^}} "-target-feature" "+c"

// RUN: %clang --target=riscv32-unknown-elf -mabi=ilp32d -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-ILP32D %s

// CHECK-ILP32D:      "-target-feature" "+m"
// CHECK-ILP32D-SAME: {{^}} "-target-feature" "+a"
// CHECK-ILP32D-SAME: {{^}} "-target-feature" "+f"
// CHECK-ILP32D-SAME: {{^}} "-target-feature" "+d"
// CHECK-ILP32D-SAME: {{^}} "-target-feature" "+c"

// RUN: %clang --target=riscv64-unknown-elf -march=rv64i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64i2p1 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64im -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64ima -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64imaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64imafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv64-unknown-elf -march=rv64ic -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64imc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64imac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64imafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64imafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv64-unknown-elf -march=rv64ia -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64iaf -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64iafd -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv64-unknown-elf -march=rv64iac -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64iafc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64iafdc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv64-unknown-elf -march=rv64g -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64gc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck %s

// RUN: %clang --target=riscv64-unknown-elf -mabi=lp64 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-LP64 %s

// CHECK-LP64: "-target-feature" "+m"
// CHECK-LP64-SAME: {{^}} "-target-feature" "+a"
// CHECK-LP64-SAME: {{^}} "-target-feature" "+f"
// CHECK-LP64-SAME: {{^}} "-target-feature" "+d"
// CHECK-LP64-SAME: {{^}} "-target-feature" "+c"

// RUN: %clang --target=riscv64-unknown-elf -mabi=lp64f -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-LP64F %s

// CHECK-LP64F: "-target-feature" "+m"
// CHECK-LP64F-SAME: {{^}} "-target-feature" "+a"
// CHECK-LP64F-SAME: {{^}} "-target-feature" "+f"
// CHECK-LP64F-SAME: {{^}} "-target-feature" "+d"
// CHECK-LP64F-SAME: {{^}} "-target-feature" "+c"

// RUN: %clang --target=riscv64-unknown-elf -mabi=lp64d -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-LP64D %s

// CHECK-LP64D: "-target-feature" "+m"
// CHECK-LP64D-SAME: {{^}} "-target-feature" "+a"
// CHECK-LP64D-SAME: {{^}} "-target-feature" "+f"
// CHECK-LP64D-SAME: {{^}} "-target-feature" "+d"
// CHECK-LP64D-SAME: {{^}} "-target-feature" "+c"

// CHECK-NOT: error: invalid arch name '

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32 %s
// RV32: error: invalid arch name 'rv32'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32m -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32M %s
// RV32M: error: invalid arch name 'rv32m'

// RUN: %clang --target=riscv32-unknown-elf -march=rv32id -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32ID %s
// RV32ID: "-target-feature" "+f" "-target-feature" "+d"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32l -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32L %s
// RV32L: error: invalid arch name 'rv32l'

// RUN: %clang --target=riscv32-unknown-elf -march=rv32imadf -### %s \
// RUN:   -fsyntax-only 2>&1 | FileCheck %s

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32imm -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32IMM %s
// RV32IMM: error: invalid arch name 'rv32imm'

// RUN: not %clang --target=riscv32-unknown-elf -march=RV32I -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32I-UPPER %s
// RV32I-UPPER: error: invalid arch name 'RV32I'

// RUN: not %clang --target=riscv64-unknown-elf -march=rv64 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64 %s
// RV64: error: invalid arch name 'rv64'

// RUN: not %clang --target=riscv64-unknown-elf -march=rv64m -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64M %s
// RV64M: error: invalid arch name 'rv64m'

// RUN: %clang --target=riscv64-unknown-elf -march=rv64id -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64ID %s
// RV64ID: "-target-feature" "+f" "-target-feature" "+d"

// RUN: not %clang --target=riscv64-unknown-elf -march=rv64l -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64L %s
// RV64L: error: invalid arch name 'rv64l'

// RUN: %clang --target=riscv64-unknown-elf -march=rv64imadf -### %s \
// RUN:   -fsyntax-only 2>&1 | FileCheck %s

// RUN: not %clang --target=riscv64-unknown-elf -march=rv64imm -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64IMM %s
// RV64IMM: error: invalid arch name 'rv64imm'

// RUN: not %clang --target=riscv64-unknown-elf -march=RV64I -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64I-UPPER %s
// RV64I-UPPER: error: invalid arch name 'RV64I'

// Testing specific messages and unsupported extensions.

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32imC -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-LOWER %s
// RV32-LOWER: error: invalid arch name 'rv32imC',
// RV32-LOWER: string must be lowercase

// RUN: not %clang --target=riscv32-unknown-elf -march=unknown -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-STR %s
// RV32-STR: error: invalid arch name 'unknown',
// RV32-STR: string must begin with rv32{i,e,g} or rv64{i,e,g}

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32q -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-LETTER %s
// RV32-LETTER: error: invalid arch name 'rv32q',
// RV32-LETTER: first letter should be 'e', 'i' or 'g'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32imcq -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ORDER %s
// RV32-ORDER: error: invalid arch name 'rv32imcq',
// RV32-ORDER: unsupported standard user-level extension 'q'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32izvl64b -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZVL64B-ER %s
// RV32-ZVL64B-ER: error: invalid arch name 'rv32izvl64b',
// RV32-ZVL64B-ER: 'zvl*b' requires 'v' or 'zve*' extension to also be specified

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32imw -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-STD-INVAL %s
// RV32-STD-INVAL: error: invalid arch name 'rv32imw',
// RV32-STD-INVAL: invalid standard user-level extension 'w'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32imqc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-STD %s
// RV32-STD: error: invalid arch name 'rv32imqc',
// RV32-STD: unsupported standard user-level extension 'q'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ib -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-B %s
// RV32-B: error: invalid arch name 'rv32ib',
// RV32-B: unsupported standard user-level extension 'b'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32xabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32X %s
// RV32X: error: invalid arch name 'rv32xabc',
// RV32X: first letter should be 'e', 'i' or 'g'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32sabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32S %s
// RV32S: error: invalid arch name 'rv32sabc',
// RV32S: first letter should be 'e', 'i' or 'g'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ix -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32X-NAME %s
// RV32X-NAME: error: invalid arch name 'rv32ix',
// RV32X-NAME: non-standard user-level extension name missing after 'x'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32is -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32S-NAME %s
// RV32S-NAME: error: invalid arch name 'rv32is',
// RV32S-NAME: standard supervisor-level extension
// RV32S-NAME: name missing after 's'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ix_s -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32ALL-NAME %s
// RV32ALL-NAME: error: invalid arch name 'rv32ix_s',
// RV32ALL-NAME: non-standard user-level extension
// RV32ALL-NAME: name missing after 'x'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ixabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32X-UNS %s
// RV32X-UNS: error: invalid arch name 'rv32ixabc',
// RV32X-UNS: unsupported non-standard user-level extension 'xabc'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32isa -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32S-UNS %s
// RV32S-UNS: error: invalid arch name 'rv32isa',
// RV32S-UNS: unsupported standard supervisor-level extension 'sa'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32isp_xabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32ALL %s
// RV32ALL: error: invalid arch name 'rv32isp_xabc',
// RV32ALL: unsupported standard supervisor-level extension 'sp'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32i20 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-IVER %s
// RV32-IVER: error: invalid arch name 'rv32i20', unsupported
// RV32-IVER: version number 20 for extension 'i'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32imc5 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-CVER %s
// RV32-CVER: error: invalid arch name 'rv32imc5', unsupported
// RV32-CVER: version number 5 for extension 'c'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32i2p -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-IMINOR-MISS %s
// RV32-IMINOR-MISS: error: invalid arch name 'rv32i2p',
// RV32-IMINOR-MISS: minor version number missing after 'p' for extension 'i'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32i2p2 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-IMINOR1 %s
// RV32-IMINOR1: error: invalid arch name 'rv32i2p2', unsupported
// RV32-IMINOR1: version number 2.2 for extension 'i'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ixt2p -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-XMINOR-MISS %s
// RV32-XMINOR-MISS: error: invalid arch name 'rv32ixt2p',
// RV32-XMINOR-MISS: minor version number missing after 'p' for extension 'xt'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ist2p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-SMINOR0 %s
// RV32-SMINOR0: error: invalid arch name 'rv32ist2p0',
// RV32-SMINOR0: unsupported version number 2.0 for extension 'st'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ixabc_ -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-XSEP %s
// RV32-XSEP: error: invalid arch name 'rv32ixabc_',
// RV32-XSEP: extension name missing after separator '_'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ixabc_a -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-PREFIX %s
// RV32-PREFIX: error: invalid arch name 'rv32ixabc_a',
// RV32-PREFIX: unsupported non-standard user-level extension 'xabc'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ixdef_sabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-X-ORDER %s
// RV32-X-ORDER: error: invalid arch name 'rv32ixdef_sabc',
// RV32-X-ORDER  unsupported non-standard user-level extension 'xdef'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ixabc_xabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-XDUP %s
// RV32-XDUP: error: invalid arch name 'rv32ixabc_xabc',
// RV32-XDUP: duplicated non-standard user-level extension 'xabc'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ixabc_xdef -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-X-X-INVAL %s
// RV32-X-X-INVAL: error: invalid arch name 'rv32ixabc_xdef', unsupported
// RV32-X-X-INVAL: non-standard user-level extension 'xabc'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32isdef_xabc -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-X-S-INVAL %s
// RV32-X-S-INVAL: error: invalid arch name 'rv32isdef_xabc',
// RV32-X-S-INVAL: unsupported standard supervisor-level extension 'sdef'

// RUN: %clang --target=riscv32-unknown-elf -march=rv32i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-TARGET %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv32i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-TARGET %s
// RUN: %clang --target=riscv32-unknown-elf -mcpu=generic-rv32 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-TARGET %s
// RUN: %clang --target=riscv64-unknown-elf -mcpu=generic-rv32 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-TARGET %s
// RV32-TARGET: "-triple" "riscv32-unknown-unknown-elf"

// RUN: %clang --target=riscv32-unknown-elf -march=rv64i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64-TARGET %s
// RUN: %clang --target=riscv64-unknown-elf -march=rv64i -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64-TARGET %s
// RUN: %clang --target=riscv32-unknown-elf -mcpu=generic-rv64 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64-TARGET %s
// RUN: %clang --target=riscv64-unknown-elf -mcpu=generic-rv64 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV64-TARGET %s
// RV64-TARGET: "-triple" "riscv64-unknown-unknown-elf"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32ifzfh01p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZFH %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32ifzfh -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZFH %s
// RV32-ZFH: "-target-feature" "+zfh"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32ifzfhmin01p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZFHMIN %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32ifzfhmin -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZFHMIN %s
// RV32-ZFHMIN: "-target-feature" "+zfhmin"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32iztso -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-NOFLAG %s
// RV32-EXPERIMENTAL-NOFLAG: error: invalid arch name 'rv32iztso'
// RV32-EXPERIMENTAL-NOFLAG: requires '-menable-experimental-extensions'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32iztso -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-NOVERS %s
// RV32-EXPERIMENTAL-NOVERS: error: invalid arch name 'rv32iztso'
// RV32-EXPERIMENTAL-NOVERS: experimental extension requires explicit version number

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32iztso0p7 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-BADVERS %s
// RV32-EXPERIMENTAL-BADVERS: error: invalid arch name 'rv32iztso0p7'
// RV32-EXPERIMENTAL-BADVERS: unsupported version number 0.7 for experimental extension 'ztso' (this compiler supports 0.1)

// RUN: %clang --target=riscv32-unknown-elf -march=rv32iztso0p1 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-GOODVERS %s
// RV32-EXPERIMENTAL-GOODVERS: "-target-feature" "+experimental-ztso"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izbb1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZBB %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32izbb -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZBB %s
// RV32-ZBB: "-target-feature" "+zbb"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32izbb1p0zbs1p0 -menable-experimental-extensions -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-EXPERIMENTAL-ZBB-ZBS-UNDERSCORE %s
// RV32-EXPERIMENTAL-ZBB-ZBS-UNDERSCORE: error: invalid arch name 'rv32izbb1p0zbs1p0', unsupported version number 1.0 for extension 'zbb1p0zbs'

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izba1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZBA %s
// RUN: %clang --target=riscv32-unknown-elf -march=rv32izba -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZBA %s
// RV32-ZBA: "-target-feature" "+zba"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32iv0p1 -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-V-BADVERS %s
// RV32-V-BADVERS: error: invalid arch name 'rv32iv0p1'
// RV32-V-BADVERS: unsupported version number 0.1 for extension 'v'

// RUN: %clang --target=riscv32-unknown-elf -march=rv32iv1p0 -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-V-GOODVERS %s
// RV32-V-GOODVERS: "-target-feature" "+v"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32iv1p0_zvl32b0p1 -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVL-BADVERS %s
// RV32-ZVL-BADVERS: error: invalid arch name 'rv32iv1p0_zvl32b0p1'
// RV32-ZVL-BADVERS: unsupported version number 0.1 for extension 'zvl32b'

// RUN: %clang --target=riscv32-unknown-elf -march=rv32iv1p0_zvl32b1p0 -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVL-GOODVERS %s
// RV32-ZVL-GOODVERS: "-target-feature" "+zvl32b"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izbkc1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZBKC %s
// RV32-ZBKC: "-target-feature" "+zbkc"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izbkx1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZBKX %s
// RV32-ZBKX: "-target-feature" "+zbkx"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izbkb1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZBKB %s
// RV32-ZBKB: "-target-feature" "+zbkb"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izknd1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZKND %s
// RV32-ZKND: "-target-feature" "+zknd"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izkne1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZKNE %s
// RV32-ZKNE: "-target-feature" "+zkne"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izknh1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZKNH %s
// RV32-ZKNH: "-target-feature" "+zknh"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izksed1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZKSED %s
// RV32-ZKSED: "-target-feature" "+zksed"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izksh1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZKSH %s
// RV32-ZKSH: "-target-feature" "+zksh"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izkr1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZKR %s
// RV32-ZKR: "-target-feature" "+zkr"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izkt1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZKT %s
// RV32-ZKT: "-target-feature" "+zkt"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izk1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZK %s
// RV32-ZK: "-target-feature" "+zk"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izfh1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-RV32-ZFH %s
// CHECK-RV32-ZFH: "-target-feature" "+zfh"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izfhmin1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=CHECK-RV32-ZFHMIN %s
// CHECK-RV32-ZFHMIN: "-target-feature" "+zfhmin"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32izve32x0p1 -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVE32X-BADVERS %s
// RV32-ZVE32X-BADVERS: error: invalid arch name 'rv32izve32x0p1'
// RV32-ZVE32X-BADVERS: unsupported version number 0.1 for extension 'zve32x'

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izve32x -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVE32X-GOODVERS %s
// RV32-ZVE32X-GOODVERS: "-target-feature" "+zve32x"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izve32f -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVE32F-GOOD %s
// RV32-ZVE32F-GOOD: "-target-feature" "+zve32f"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izve64x -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVE64X %s
// RV32-ZVE64X: "-target-feature" "+zve64x"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izve64f -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVE64F-GOOD %s
// RV32-ZVE64F-GOOD: "-target-feature" "+zve64f"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izve64d -### %s -c 2>&1 | \
// RUN:   FileCheck -check-prefix=RV32-ZVE64D-GOOD %s
// RV32-ZVE64D-GOOD: "-target-feature" "+zve64d"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izfinx -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZFINX-GOOD %s
// RV32-ZFINX-GOOD: "-target-feature" "+zfinx"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izdinx -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZDINX-GOOD %s
// RV32-ZDINX-GOOD: "-target-feature" "+zdinx"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izhinxmin -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZHINXMIN-GOOD %s
// RV32-ZHINXMIN-GOOD: "-target-feature" "+zhinxmin"

// RUN: %clang --target=riscv32-unknown-elf -march=rv32izhinx1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZHINX-GOOD %s
// RV32-ZHINX-GOOD: "-target-feature" "+zhinx"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32izhinx0p1 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZHINX-BADVERS %s
// RV32-ZHINX-BADVERS: error: invalid arch name 'rv32izhinx0p1'
// RV32-ZHINX-BADVERS: unsupported version number 0.1 for extension 'zhinx'

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32i_zmmul2p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZMMUL-BADVERS %s
// RV32-ZMMUL-BADVERS: error: invalid arch name 'rv32i_zmmul2p0'
// RV32-ZMMUL-BADVERS: unsupported version number 2.0 for extension

// RUN: %clang --target=riscv32-unknown-elf -march=rv32i_zmmul1p0 -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-ZMMUL-GOODVERS %s
// RV32-ZMMUL-GOODVERS: "-target-feature" "+zmmul"

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32ifzfinx -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-F-ZFINX-ER %s
// RV32-F-ZFINX-ER: error: invalid arch name 'rv32ifzfinx',
// RV32-F-ZFINX-ER: 'f' and 'zfinx' extensions are incompatible

// RUN: not %clang --target=riscv32-unknown-elf -march=rv32idzdinx -### %s \
// RUN: -fsyntax-only 2>&1 | FileCheck -check-prefix=RV32-D-ZDINX-ER %s
// RV32-D-ZDINX-ER: error: invalid arch name 'rv32idzdinx',
// RV32-D-ZFINX-ER: 'f' and 'zfinx' extensions are incompatible
