// REQUIRES: x86-registered-target
// RUN: %clang --target=x86_64-unknown-linux-gnu -c %s -o /dev/null

// Check that cc1as can use assembler info in object generation.
.data
	
foo:
.if . - foo == 0
    .byte 0xaa
.else
    .byte 0x00
.endif
