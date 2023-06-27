// RUN: %clang_cc1 -triple thumbv7-apple-darwin10 -target-cpu cortex-a8 -fsyntax-only -Wvector-conversions -ffreestanding %s
// RUN: %clang_cc1 -triple thumbv7-apple-darwin10 -target-cpu cortex-a8 -fsyntax-only -flax-vector-conversions=none -ffreestanding %s
// RUN: %clang_cc1 -x c++ -triple thumbv7-apple-darwin10 -target-cpu cortex-a8 -fsyntax-only -Wvector-conversions -ffreestanding %s

// RUN: %clang -fsyntax-only               -ffreestanding --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c89 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c99 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c11 -xc %s

// RUN: %clang -fsyntax-only               -ffreestanding --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c89 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c99 -xc %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c11 -xc %s

// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++98 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++11 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++14 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64-none-elf -march=armv8.2-a+fp16 -std=c++17 -xc++ %s

// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++98 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++11 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++14 -xc++ %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding -nostdinc++ --target=aarch64_be-none-elf -march=armv8.2-a+fp16 -std=c++17 -xc++ %s

// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64-none-elf -march=armv8.2-a+fp16fml+crypto+dotprod -std=c11 -xc -flax-vector-conversions=none %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=aarch64_be-none-elf -march=armv8.2-a+fp16fml+crypto+dotprod -std=c11 -xc -flax-vector-conversions=none %s
// RUN: %clang -fsyntax-only -Wall -Werror -ffreestanding --target=arm64-linux-gnu -arch +neon -std=c11 -xc -flax-vector-conversions=none %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>
