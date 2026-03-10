// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

void general(unsigned flags, bool value) {
    flags |= value;

    unsigned mask = 0b1100;
    bool result = flags &= mask;
    auto result2 = flags &= mask;
    result = flags |= flags << 1;
}

void assign_to_boolean(unsigned flags, bool value) {
    value |= flags << 1;
    value |= (flags << 1) | (flags << 2);
}
