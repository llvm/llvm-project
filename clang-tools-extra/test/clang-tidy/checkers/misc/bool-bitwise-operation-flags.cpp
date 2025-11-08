// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

bool normal() {
    unsigned flags = 100;
    if (flags & 0xFFFFFFF8) {

    }
    return flags & 0x20;
}
