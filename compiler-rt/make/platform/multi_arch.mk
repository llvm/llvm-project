Description := Example configuration for build two libraries for separate \
architectures.

Configs := m32 m64
Arch := i386
Arch.m64 := x86_64

CC := gcc
CC.m32 := clang

CFLAGS := -Wall -Werror
CFLAGS.m32 := $(CFLAGS) -m32 -O3
CFLAGS.m64 := $(CFLAGS) -m64 -O3

FUNCTIONS := moddi3 floatundixf udivdi3
FUNCTIONS.m64 := $(FUNCTIONS) lshrdi3
