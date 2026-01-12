// RUN: %clang -fsyntax-only  -ffreestanding              --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c89 --sysroot=%S/Inputs -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c99 --sysroot=%S/Inputs -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c11 --sysroot=%S/Inputs -xc %s

// RUN: %clang -fsyntax-only -ffreestanding               --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c89 --sysroot=%S/Inputs -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c99 --sysroot=%S/Inputs -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c11 --sysroot=%S/Inputs -xc %s

// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++98 --sysroot=%S/Inputs -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++11 --sysroot=%S/Inputs -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++14 --sysroot=%S/Inputs -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++17 --sysroot=%S/Inputs -xc++ %s

// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++98 --sysroot=%S/Inputs -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++11 --sysroot=%S/Inputs -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++14 --sysroot=%S/Inputs -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++17 --sysroot=%S/Inputs -xc++ %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include "system_reserved_names.h"

#include <arm_fp16.h>
