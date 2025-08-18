// Check target CPUs are correctly passed.

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=rocket-rv32 | FileCheck -check-prefix=MCPU-ROCKET32 %s
// MCPU-ROCKET32: "-nostdsysteminc" "-target-cpu" "rocket-rv32"
// MCPU-ROCKET32: "-target-feature" "+zicsr" "-target-feature" "+zifencei"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=rocket-rv64 | FileCheck -check-prefix=MCPU-ROCKET64 %s
// MCPU-ROCKET64: "-nostdsysteminc" "-target-cpu" "rocket-rv64"
// MCPU-ROCKET64: "-target-feature" "+zicsr" "-target-feature" "+zifencei"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=syntacore-scr1-base | FileCheck -check-prefix=MCPU-SYNTACORE-SCR1-BASE %s
// MCPU-SYNTACORE-SCR1-BASE: "-target-cpu" "syntacore-scr1-base"
// MCPU-SYNTACORE-SCR1-BASE: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR1-BASE: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR1-BASE: "-target-abi" "ilp32"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=syntacore-scr1-max | FileCheck -check-prefix=MCPU-SYNTACORE-SCR1-MAX %s
// MCPU-SYNTACORE-SCR1-MAX: "-target-cpu" "syntacore-scr1-max"
// MCPU-SYNTACORE-SCR1-MAX: "-target-feature" "+m" "-target-feature" "+c"
// MCPU-SYNTACORE-SCR1-MAX: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR1-MAX: "-target-abi" "ilp32"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=xiangshan-nanhu | FileCheck -check-prefix=MCPU-XIANGSHAN-NANHU %s
// MCPU-XIANGSHAN-NANHU: "-nostdsysteminc" "-target-cpu" "xiangshan-nanhu"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-feature" "+c"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-feature" "+zicbom" "-target-feature" "+zicboz" "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-feature" "+zba" "-target-feature" "+zbb" "-target-feature" "+zbc"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-feature" "+zbkb" "-target-feature" "+zbkc" "-target-feature" "+zbkx" "-target-feature" "+zbs"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-feature" "+zkn" "-target-feature" "+zknd" "-target-feature" "+zkne" "-target-feature" "+zknh"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-feature" "+zks" "-target-feature" "+zksed" "-target-feature" "+zksh" "-target-feature" "+svinval"
// MCPU-XIANGSHAN-NANHU-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=xiangshan-kunminghu | FileCheck -check-prefix=MCPU-XIANGSHAN-KUNMINGHU %s
// MCPU-XIANGSHAN-KUNMINGHU: "-nostdsysteminc" "-target-cpu" "xiangshan-kunminghu"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+m"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+a"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+f"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+d"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+c"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+b"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+v"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+h"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zic64b" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zicbom" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zicbop" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zicboz" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ziccamoa" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ziccif" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zicclsm" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ziccrse" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zicntr" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zicond" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zicsr" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zihintntl" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zacas" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zawrs" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zfa" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zfh" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zca" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zcb" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zcmop" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zba" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zbb" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zbc"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zbkb" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zbkc" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zbkx" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zbs"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zkn" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zks" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zvbb" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zve64d" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zve64f" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zve64x" 
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zvfh"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zvkb"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zvkt"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+zvl128b"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+sha"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+shcounterenw"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+shgatpa"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+shtvala"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+shvsatpa"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+shvstvala"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+shvstvecd"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+smcsrind"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+smdbltrp"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+smmpm"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+smnpm"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+smrnmi"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+smstateen"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+sscofpmf"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+sscsrind"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ssdbltrp"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ssnpm"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+sspm"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ssstateen"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ssstrict"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+sstc"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+ssu64xl"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+supm"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-feature" "+svnapot"
// MCPU-XIANGSHAN-KUNMINGHU-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=spacemit-x60 | FileCheck -check-prefix=MCPU-SPACEMIT-X60 %s
// MCPU-SPACEMIT-X60: "-nostdsysteminc" "-target-cpu" "spacemit-x60"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+m"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+a"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+f"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+d"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+c"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+v"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zic64b"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zicbom"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zicbop"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zicboz"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+ziccamoa"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+ziccif"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zicclsm"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+ziccrse"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zicntr"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zicond"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zicsr"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zifencei"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zihintpause"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zihpm"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+za64rs"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zfh"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zfhmin"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zba"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zbb"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zbc"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zbkc"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zbs"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zkt"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zve32f"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zve32x"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zve64d"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zve64f"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zve64x"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zvfh"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zvfhmin"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zvkt"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zvl128b"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zvl256b"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zvl32b"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+zvl64b"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+ssccptr"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+sscofpmf"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+sscounterenw"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+sstc"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+sstvala"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+sstvecd"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+svade"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+svbare"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+svinval"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+svnapot"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+svpbmt"
// MCPU-SPACEMIT-X60-SAME: "-target-feature" "+xsmtvdot"
// MCPU-SPACEMIT-X60-SAME: "-target-abi" "lp64d"

// We cannot check much for -mcpu=native, but it should be replaced by a valid CPU string.
// RUN: %clang --target=riscv64 -### -c %s -mcpu=native 2> %t.err || true
// RUN: FileCheck --input-file=%t.err -check-prefix=MCPU-NATIVE %s
// MCPU-NATIVE-NOT: "-target-cpu" "native"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=rocket-rv32 | FileCheck -check-prefix=MTUNE-ROCKET32 %s
// MTUNE-ROCKET32: "-tune-cpu" "rocket-rv32"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=rocket-rv64 | FileCheck -check-prefix=MTUNE-ROCKET64 %s
// MTUNE-ROCKET64: "-tune-cpu" "rocket-rv64"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=mips-p8700 | FileCheck -check-prefix=MTUNE-MIPS-P8700 %s
// MTUNE-MIPS-P8700: "-tune-cpu" "mips-p8700"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=mips-p8700 | FileCheck -check-prefix=MCPU-MIPS-P8700 %s
// MCPU-MIPS-P8700: "-target-cpu" "mips-p8700"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+m"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+a"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+f"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+d"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+c"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+zicsr"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+zifencei"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+zaamo"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+zalrsc"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+zba"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+zbb"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+xmipscbop"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+xmipscmov"
// MCPU-MIPS-P8700-SAME: "-target-feature" "+xmipslsp"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=syntacore-scr1-base | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR1-BASE %s
// MTUNE-SYNTACORE-SCR1-BASE: "-tune-cpu" "syntacore-scr1-base"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=syntacore-scr1-max | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR1-MAX %s
// MTUNE-SYNTACORE-SCR1-MAX: "-tune-cpu" "syntacore-scr1-max"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=tt-ascalon-d8 | FileCheck -check-prefix=MTUNE-TT-ASCALON-D8 %s
// MTUNE-TT-ASCALON-D8: "-tune-cpu" "tt-ascalon-d8"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=tt-ascalon-d8 | FileCheck -check-prefix=MCPU-TT-ASCALON-D8 %s
// MCPU-TT-ASCALON-D8: "-target-cpu" "tt-ascalon-d8"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+m"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+a"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+f"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+d"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+c"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+v"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+h"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zicbom"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zicbop"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zicboz"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zicntr"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zicond"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zicsr"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zifencei"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zihintntl"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zihintpause"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zihpm"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zimop"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zmmul"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zawrs"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zfa"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zfbfmin"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zfh"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zfhmin"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zca"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zcb"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zba"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zbb"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zbs"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zkt"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvbb"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvbc"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zve32f"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zve32x"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zve64d"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zve64f"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zve64x"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvfbfmin"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvfbfwma"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvfh"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvfhmin"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvkb"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvkg"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvkn"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvknc"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvkned"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvkng"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvknhb"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvkt"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvl128b"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvl256b"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvl32b"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+zvl64b"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+sscofpmf"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+svinval"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+svnapot"
// MCPU-TT-ASCALON-D8-SAME: "-target-feature" "+svpbmt"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=veyron-v1 | FileCheck -check-prefix=MCPU-VEYRON-V1 %s
// MCPU-VEYRON-V1: "-target-cpu" "veyron-v1"
// MCPU-VEYRON-V1: "-target-feature" "+m"
// MCPU-VEYRON-V1: "-target-feature" "+a"
// MCPU-VEYRON-V1: "-target-feature" "+f"
// MCPU-VEYRON-V1: "-target-feature" "+d"
// MCPU-VEYRON-V1: "-target-feature" "+c"
// MCPU-VEYRON-V1: "-target-feature" "+zicbom"
// MCPU-VEYRON-V1: "-target-feature" "+zicbop"
// MCPU-VEYRON-V1: "-target-feature" "+zicboz"
// MCPU-VEYRON-V1: "-target-feature" "+zicntr"
// MCPU-VEYRON-V1: "-target-feature" "+zicsr"
// MCPU-VEYRON-V1: "-target-feature" "+zifencei"
// MCPU-VEYRON-V1: "-target-feature" "+zihintpause"
// MCPU-VEYRON-V1: "-target-feature" "+zihpm"
// MCPU-VEYRON-V1: "-target-feature" "+zba"
// MCPU-VEYRON-V1: "-target-feature" "+zbb"
// MCPU-VEYRON-V1: "-target-feature" "+zbc"
// MCPU-VEYRON-V1: "-target-feature" "+zbs"
// MCPU-VEYRON-V1: "-target-feature" "+xventanacondops"
// MCPU-VEYRON-V1: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=veyron-v1 | FileCheck -check-prefix=MTUNE-VEYRON-V1 %s
// MTUNE-VEYRON-V1: "-tune-cpu" "veyron-v1"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=xiangshan-nanhu | FileCheck -check-prefix=MTUNE-XIANGSHAN-NANHU %s
// MTUNE-XIANGSHAN-NANHU: "-tune-cpu" "xiangshan-nanhu"

// Check -mtune alias CPU has resolved to the right CPU according XLEN.
// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=generic | FileCheck -check-prefix=MTUNE-GENERIC-32 %s
// MTUNE-GENERIC-32: "-tune-cpu" "generic"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=generic | FileCheck -check-prefix=MTUNE-GENERIC-64 %s
// MTUNE-GENERIC-64: "-tune-cpu" "generic"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=rocket | FileCheck -check-prefix=MTUNE-ROCKET-32 %s
// MTUNE-ROCKET-32: "-tune-cpu" "rocket"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=rocket | FileCheck -check-prefix=MTUNE-ROCKET-64 %s
// MTUNE-ROCKET-64: "-tune-cpu" "rocket"

// We cannot check much for -mtune=native, but it should be replaced by a valid CPU string.
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=native | FileCheck -check-prefix=MTUNE-NATIVE %s
// MTUNE-NATIVE-NOT: "-tune-cpu" "native"

// -mcpu with default -march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e20 | FileCheck -check-prefix=MCPU-SIFIVE-E20 %s
// MCPU-SIFIVE-E20: "-nostdsysteminc" "-target-cpu" "sifive-e20"
// MCPU-SIFIVE-E20: "-target-feature" "+m" "-target-feature" "+c"
// MCPU-SIFIVE-E20: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-E20: "-target-abi" "ilp32"

// -mcpu with default -march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e21 | FileCheck -check-prefix=MCPU-SIFIVE-E21 %s
// MCPU-SIFIVE-E21: "-nostdsysteminc" "-target-cpu" "sifive-e21"
// MCPU-SIFIVE-E21: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+c"
// MCPU-SIFIVE-E21: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-E21: "-target-abi" "ilp32"

// -mcpu with default -march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e24 | FileCheck -check-prefix=MCPU-SIFIVE-E24 %s
// MCPU-SIFIVE-E24: "-nostdsysteminc" "-target-cpu" "sifive-e24"
// MCPU-SIFIVE-E24: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E24: "-target-feature" "+c"
// MCPU-SIFIVE-E24: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-E24: "-target-abi" "ilp32f"

// -mcpu with default -march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e34 | FileCheck -check-prefix=MCPU-SIFIVE-E34 %s
// MCPU-SIFIVE-E34: "-nostdsysteminc" "-target-cpu" "sifive-e34"
// MCPU-SIFIVE-E34: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E34: "-target-feature" "+c"
// MCPU-SIFIVE-E34: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-E34: "-target-abi" "ilp32f"

// -mcpu with -mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s21 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-S21 %s
// MCPU-ABI-SIFIVE-S21: "-nostdsysteminc" "-target-cpu" "sifive-s21"
// MCPU-ABI-SIFIVE-S21: "-target-feature" "+m" "-target-feature" "+a"
// MCPU-ABI-SIFIVE-S21: "-target-feature" "+c"
// MCPU-ABI-SIFIVE-S21: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-ABI-SIFIVE-S21: "-target-abi" "lp64"

// -mcpu with -mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s51 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-S51 %s
// MCPU-ABI-SIFIVE-S51: "-nostdsysteminc" "-target-cpu" "sifive-s51"
// MCPU-ABI-SIFIVE-S51: "-target-feature" "+m" "-target-feature" "+a"
// MCPU-ABI-SIFIVE-S51: "-target-feature" "+c"
// MCPU-ABI-SIFIVE-S51: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-ABI-SIFIVE-S51: "-target-abi" "lp64"

// -mcpu with default -march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s54 | FileCheck -check-prefix=MCPU-SIFIVE-S54 %s
// MCPU-SIFIVE-S54: "-nostdsysteminc" "-target-cpu" "sifive-s54"
// MCPU-SIFIVE-S54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-S54: "-target-feature" "+c"
// MCPU-SIFIVE-S54: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-S54: "-target-abi" "lp64d"

// -mcpu with -mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-s76 | FileCheck -check-prefix=MCPU-SIFIVE-S76 %s
// MCPU-SIFIVE-S76: "-nostdsysteminc" "-target-cpu" "sifive-s76"
// MCPU-SIFIVE-S76: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-S76: "-target-feature" "+c"
// MCPU-SIFIVE-S76: "-target-feature" "+zicsr" "-target-feature" "+zifencei" "-target-feature" "+zihintpause"
// MCPU-SIFIVE-S76: "-target-abi" "lp64d"

// -mcpu with default -march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-u54 | FileCheck -check-prefix=MCPU-SIFIVE-U54 %s
// MCPU-SIFIVE-U54: "-nostdsysteminc" "-target-cpu" "sifive-u54"
// MCPU-SIFIVE-U54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-U54: "-target-feature" "+c"
// MCPU-SIFIVE-U54: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-U54: "-target-abi" "lp64d"

// -mcpu with -mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-u54 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-U54 %s
// MCPU-ABI-SIFIVE-U54: "-nostdsysteminc" "-target-cpu" "sifive-u54"
// MCPU-ABI-SIFIVE-U54: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-ABI-SIFIVE-U54: "-target-feature" "+c"
// MCPU-ABI-SIFIVE-U54: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-ABI-SIFIVE-U54: "-target-abi" "lp64"

// -mcpu with default -march
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-e76 | FileCheck -check-prefix=MCPU-SIFIVE-E76 %s
// MCPU-SIFIVE-E76: "-nostdsysteminc" "-target-cpu" "sifive-e76"
// MCPU-SIFIVE-E76: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f"
// MCPU-SIFIVE-E76: "-target-feature" "+c"
// MCPU-SIFIVE-E76: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-E76: "-target-abi" "ilp32f"

// -mcpu with -mabi option
// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=sifive-u74 -mabi=lp64 | FileCheck -check-prefix=MCPU-ABI-SIFIVE-U74 %s
// MCPU-ABI-SIFIVE-U74: "-nostdsysteminc" "-target-cpu" "sifive-u74"
// MCPU-ABI-SIFIVE-U74: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-ABI-SIFIVE-U74: "-target-feature" "+c"
// MCPU-ABI-SIFIVE-U74: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-ABI-SIFIVE-U74: "-target-abi" "lp64"

// -march overwrite -mcpu's default -march
// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=sifive-e31 -march=rv32imc | FileCheck -check-prefix=MCPU-MARCH %s
// MCPU-MARCH: "-nostdsysteminc" "-target-cpu" "sifive-e31" "-target-feature" "+m" "-target-feature" "+c"
// MCPU-MARCH: "-target-abi" "ilp32"

// -march=unset erases previous march
// RUN: %clang --target=riscv32 -### -c %s 2>&1 -march=rv32imc -march=unset -mcpu=sifive-e31 | FileCheck -check-prefix=MARCH-UNSET %s
// MARCH-UNSET: "-nostdsysteminc" "-target-cpu" "sifive-e31" "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+c"
// MARCH-UNSET-SAME: "-target-abi" "ilp32"

// Check interaction between -mcpu and mtune, -mtune won't affect arch related
// target feature, but -mcpu will.
//
// In this case, sifive-e31 is rv32imac, sifive-e76 is rv32imafc, so F-extension
// should not enabled.
//
// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=sifive-e31 -mtune=sifive-e76 | FileCheck -check-prefix=MTUNE-E31-MCPU-E76 %s
// MTUNE-E31-MCPU-E76: "-target-cpu" "sifive-e31"
// MTUNE-E31-MCPU-E76-NOT: "-target-feature" "+f"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+m"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+a"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+c"
// MTUNE-E31-MCPU-E76-SAME: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MTUNE-E31-MCPU-E76-SAME: "-tune-cpu" "sifive-e76"

// -mcpu with default -march include experimental extensions
// RUN: %clang -target riscv64 -### -c %s 2>&1 -menable-experimental-extensions -mcpu=sifive-x280 | FileCheck -check-prefix=MCPU-SIFIVE-X280 %s
// MCPU-SIFIVE-X280: "-nostdsysteminc" "-target-cpu" "sifive-x280"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+c" "-target-feature" "+v"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zicsr" "-target-feature" "+zifencei"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zfh"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zba" "-target-feature" "+zbb"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvfh"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl128b"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl256b" "-target-feature" "+zvl32b"
// MCPU-SIFIVE-X280-SAME: "-target-feature" "+zvl512b" "-target-feature" "+zvl64b"
// MCPU-SIFIVE-X280-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -menable-experimental-extensions -mcpu=sifive-x390 | FileCheck -check-prefix=MCPU-SIFIVE-X390 %s
// MCPU-SIFIVE-X390: "-target-cpu" "sifive-x390"
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-sifive-x390.c`
// MCPU-SIFIVE-X390-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p450 | FileCheck -check-prefix=MCPU-SIFIVE-P450 %s
// MCPU-SIFIVE-P450: "-nostdsysteminc" "-target-cpu" "sifive-p450"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+m"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+a"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+f"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+d"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+c"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zic64b"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zicbom"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zicbop"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zicboz"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+ziccamoa"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+ziccif"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zicclsm"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+ziccrse"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zicntr"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zicsr"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zifencei"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zihintntl"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zihintpause"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zihpm"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+za64rs"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zfhmin"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zba"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zbb"
// MCPU-SIFIVE-P450-SAME: "-target-feature" "+zbs"
// MCPU-SIFIVE-P450-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p470 | FileCheck -check-prefix=MCPU-SIFIVE-P470 %s
// MCPU-SIFIVE-P470: "-target-cpu" "sifive-p470"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+m"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+a"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+f"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+d"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+c"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+v"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zic64b"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zicbom"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zicbop"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zicboz"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+ziccamoa"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+ziccif"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zicclsm"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+ziccrse"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zicntr"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zicsr"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zifencei"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zihintntl"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zihintpause"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zihpm"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zmmul"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+za64rs"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zfhmin"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zba"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zbb"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zbs"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvbb"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvbc"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zve32f"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zve32x"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zve64d"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zve64f"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zve64x"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvkg"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvkn"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvknc"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvkned"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvkng"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvknhb"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvks"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvksc"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvksed"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvksg"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvksh"
// MCPU-SIFIVE-P470-SAME: "-target-feature" "+zvkt"
// MCPU-SIFIVE-P470-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p550 | FileCheck -check-prefix=MCPU-SIFIVE-P550 %s
// MCPU-SIFIVE-P550: "-nostdsysteminc" "-target-cpu" "sifive-p550"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+m"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+a"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+f"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+d"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+c"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+zicsr"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+zifencei"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+zba"
// MCPU-SIFIVE-P550-SAME: "-target-feature" "+zbb"
// MCPU-SIFIVE-P550-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p670 | FileCheck -check-prefix=MCPU-SIFIVE-P670 %s
// MCPU-SIFIVE-P670: "-target-cpu" "sifive-p670"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+m"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+a"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+f"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+d"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+c"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+v"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zic64b"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zicbom"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zicbop"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zicboz"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+ziccamoa"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+ziccif"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zicclsm"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+ziccrse"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zicntr"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zicsr"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zifencei"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zihintntl"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zihintpause"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zihpm"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+za64rs"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zfhmin"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zba"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zbb"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zbs"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvbb"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvbc"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zve32f"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zve32x"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zve64d"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zve64f"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zve64x"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvkg"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvkn"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvknc"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvkned"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvkng"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvknhb"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvks"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvksc"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvksed"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvksg"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvksh"
// MCPU-SIFIVE-P670-SAME: "-target-feature" "+zvkt"
// MCPU-SIFIVE-P670-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv64 -### -c %s 2>&1 -mcpu=sifive-p870 | FileCheck -check-prefix=MCPU-SIFIVE-P870 %s
// MCPU-SIFIVE-P870: "-target-cpu" "sifive-p870"
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-sifive-p870.c`
// MCPU-SIFIVE-P870-SAME: "-target-abi" "lp64d"

// RUN: %clang -target riscv32 -### -c %s 2>&1 -mcpu=rp2350-hazard3 | FileCheck -check-prefix=MCPU-HAZARD3 %s
// MCPU-HAZARD3: "-target-cpu" "rp2350-hazard3"
// MCPU-HAZARD3-SAME: "-target-feature" "+m"
// MCPU-HAZARD3-SAME: "-target-feature" "+a"
// MCPU-HAZARD3-SAME: "-target-feature" "+c"
// MCPU-HAZARD3-SAME: "-target-feature" "+zicsr"
// MCPU-HAZARD3-SAME: "-target-feature" "+zifencei"
// MCPU-HAZARD3-SAME: "-target-feature" "+zcb"
// MCPU-HAZARD3-SAME: "-target-feature" "+zcmp"
// MCPU-HAZARD3-SAME: "-target-feature" "+zba"
// MCPU-HAZARD3-SAME: "-target-feature" "+zbb"
// MCPU-HAZARD3-SAME: "-target-feature" "+zbkb"
// MCPU-HAZARD3-SAME: "-target-feature" "+zbs"
// MCPU-HAZARD3-SAME: "-target-abi" "ilp32"

// Check failed cases

// RUN: not %clang --target=riscv32 -### -c %s 2>&1 -mcpu=generic-rv321 | FileCheck -check-prefix=FAIL-MCPU-NAME %s
// FAIL-MCPU-NAME: error: unsupported argument 'generic-rv321' to option '-mcpu='

// RUN: not %clang --target=riscv32 -### -c %s 2>&1 -mcpu=generic-rv32 -march=rv64i | FileCheck -check-prefix=MISMATCH-ARCH %s
// MISMATCH-ARCH: cpu 'generic-rv32' does not support rv64

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=syntacore-scr3-rv32 | FileCheck -check-prefix=MCPU-SYNTACORE-SCR3-RV32 %s
// MCPU-SYNTACORE-SCR3-RV32: "-target-cpu" "syntacore-scr3-rv32"
// MCPU-SYNTACORE-SCR3-RV32-SAME: "-target-feature" "+m"
// MCPU-SYNTACORE-SCR3-RV32-SAME: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR3-RV32-SAME: "-target-feature" "+zicsr"
// MCPU-SYNTACORE-SCR3-RV32-SAME: "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR3-RV32-SAME: "-target-abi" "ilp32"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=syntacore-scr3-rv32 | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR3-RV32 %s
// MTUNE-SYNTACORE-SCR3-RV32: "-tune-cpu" "syntacore-scr3-rv32"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=syntacore-scr3-rv64 | FileCheck -check-prefix=MCPU-SYNTACORE-SCR3-RV64 %s
// MCPU-SYNTACORE-SCR3-RV64: "-target-cpu" "syntacore-scr3-rv64"
// MCPU-SYNTACORE-SCR3-RV64-SAME: "-target-feature" "+m"
// MCPU-SYNTACORE-SCR3-RV64-SAME: "-target-feature" "+a"
// MCPU-SYNTACORE-SCR3-RV64-SAME: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR3-RV64-SAME: "-target-feature" "+zicsr"
// MCPU-SYNTACORE-SCR3-RV64-SAME: "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR3-RV64-SAME: "-target-abi" "lp64"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=syntacore-scr3-rv64 | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR3-RV64 %s
// MTUNE-SYNTACORE-SCR3-RV64: "-tune-cpu" "syntacore-scr3-rv64"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=syntacore-scr4-rv32 | FileCheck -check-prefix=MCPU-SYNTACORE-SCR4-RV32 %s
// MCPU-SYNTACORE-SCR4-RV32: "-target-cpu" "syntacore-scr4-rv32"
// MCPU-SYNTACORE-SCR4-RV32-SAME: "-target-feature" "+m"
// MCPU-SYNTACORE-SCR4-RV32-SAME: "-target-feature" "+f"
// MCPU-SYNTACORE-SCR4-RV32-SAME: "-target-feature" "+d"
// MCPU-SYNTACORE-SCR4-RV32-SAME: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR4-RV32-SAME: "-target-feature" "+zicsr"
// MCPU-SYNTACORE-SCR4-RV32-SAME: "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR4-RV32-SAME: "-target-abi" "ilp32d"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=syntacore-scr4-rv32 | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR4-RV32 %s
// MTUNE-SYNTACORE-SCR4-RV32: "-tune-cpu" "syntacore-scr4-rv32"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=syntacore-scr4-rv64 | FileCheck -check-prefix=MCPU-SYNTACORE-SCR4-RV64 %s
// MCPU-SYNTACORE-SCR4-RV64: "-target-cpu" "syntacore-scr4-rv64"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-feature" "+m"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-feature" "+a"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-feature" "+f"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-feature" "+d"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-feature" "+zicsr"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR4-RV64-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=syntacore-scr4-rv64 | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR4-RV64 %s
// MTUNE-SYNTACORE-SCR4-RV64: "-tune-cpu" "syntacore-scr4-rv64"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=syntacore-scr5-rv32 | FileCheck -check-prefix=MCPU-SYNTACORE-SCR5-RV32 %s
// MCPU-SYNTACORE-SCR5-RV32: "-target-cpu" "syntacore-scr5-rv32"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-feature" "+m"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-feature" "+a"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-feature" "+f"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-feature" "+d"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-feature" "+zicsr"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR5-RV32-SAME: "-target-abi" "ilp32d"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=syntacore-scr5-rv32 | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR5-RV32 %s
// MTUNE-SYNTACORE-SCR5-RV32: "-tune-cpu" "syntacore-scr5-rv32"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=syntacore-scr5-rv64 | FileCheck -check-prefix=MCPU-SYNTACORE-SCR5-RV64 %s
// MCPU-SYNTACORE-SCR5-RV64: "-target-cpu" "syntacore-scr5-rv64"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-feature" "+m"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-feature" "+a"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-feature" "+f"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-feature" "+d"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-feature" "+zicsr"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR5-RV64-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=syntacore-scr5-rv64 | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR5-RV64 %s
// MTUNE-SYNTACORE-SCR5-RV64: "-tune-cpu" "syntacore-scr5-rv64"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=syntacore-scr7 | FileCheck -check-prefix=MCPU-SYNTACORE-SCR7 %s
// MCPU-SYNTACORE-SCR7: "-target-cpu" "syntacore-scr7"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+m"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+a"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+f"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+d"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+c"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+v"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zicsr"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zifencei"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zba"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zbb"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zbc"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zbkb"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zbkc"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zbkx"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zbs"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zkn"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zknd"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zkne"
// MCPU-SYNTACORE-SCR7-SAME: "-target-feature" "+zknh"
// MCPU-SYNTACORE-SCR7-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=syntacore-scr7 | FileCheck -check-prefix=MTUNE-SYNTACORE-SCR7 %s
// MTUNE-SYNTACORE-SCR7: "-tune-cpu" "syntacore-scr7"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=andes-a25 | FileCheck -check-prefix=MCPU-ANDES-A25 %s
// MCPU-ANDES-A25: "-target-cpu" "andes-a25"
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-andes-a25.c`
// MCPU-ANDES-A25-SAME: "-target-abi" "ilp32d"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=andes-a25 | FileCheck -check-prefix=MTUNE-ANDES-A25 %s
// MTUNE-ANDES-A25: "-tune-cpu" "andes-a25"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=andes-ax25 | FileCheck -check-prefix=MCPU-ANDES-AX25 %s
// MCPU-ANDES-AX25: "-target-cpu" "andes-ax25"
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-andes-ax25.c`
// MCPU-ANDES-AX25-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=andes-ax25 | FileCheck -check-prefix=MTUNE-ANDES-AX25 %s
// MTUNE-ANDES-AX25: "-tune-cpu" "andes-ax25"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=andes-n45 | FileCheck -check-prefix=MCPU-ANDES-N45 %s
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-andes-n45.c`
// MCPU-ANDES-N45: "-target-cpu" "andes-n45"
// MCPU-ANDES-N45-SAME: "-target-abi" "ilp32d"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=andes-n45 | FileCheck -check-prefix=MTUNE-ANDES-N45 %s
// MTUNE-ANDES-N45: "-tune-cpu" "andes-n45"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=andes-nx45 | FileCheck -check-prefix=MCPU-ANDES-NX45 %s
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-andes-nx45.c`
// MCPU-ANDES-NX45: "-target-cpu" "andes-nx45"
// MCPU-ANDES-NX45-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=andes-nx45 | FileCheck -check-prefix=MTUNE-ANDES-NX45 %s
// MTUNE-ANDES-NX45: "-tune-cpu" "andes-nx45"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mcpu=andes-a45 | FileCheck -check-prefix=MCPU-ANDES-A45 %s
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-andes-a45.c`
// MCPU-ANDES-A45: "-target-cpu" "andes-a45"
// MCPU-ANDES-A45-SAME: "-target-abi" "ilp32d"

// RUN: %clang --target=riscv32 -### -c %s 2>&1 -mtune=andes-a45 | FileCheck -check-prefix=MTUNE-ANDES-A45 %s
// MTUNE-ANDES-A45: "-tune-cpu" "andes-a45"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=andes-ax45 | FileCheck -check-prefix=MCPU-ANDES-AX45 %s
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-andes-ax45.c`
// MCPU-ANDES-AX45: "-target-cpu" "andes-ax45"
// MCPU-ANDES-AX45-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=andes-ax45 | FileCheck -check-prefix=MTUNE-ANDES-AX45 %s
// MTUNE-ANDES-AX45: "-tune-cpu" "andes-ax45"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mcpu=andes-ax45mpv | FileCheck -check-prefix=MCPU-ANDES-AX45MPV %s
// COM: The list of extensions are tested in `test/Driver/print-enabled-extensions/riscv-andes-ax45mpv.c`
// MCPU-ANDES-AX45MPV: "-target-cpu" "andes-ax45mpv"
// MCPU-ANDES-AX45MPV-SAME: "-target-abi" "lp64d"

// RUN: %clang --target=riscv64 -### -c %s 2>&1 -mtune=andes-ax45mpv | FileCheck -check-prefix=MTUNE-ANDES-AX45MPV %s
// MTUNE-ANDES-AX45MPV: "-tune-cpu" "andes-ax45mpv"
