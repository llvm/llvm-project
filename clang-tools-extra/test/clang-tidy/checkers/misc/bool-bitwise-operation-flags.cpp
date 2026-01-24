// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

bool normal() {
    unsigned flags = 100;
    if (flags & 0xFFFFFFF8) {

    }
    if (flags | 0x0) {

    }
    return (flags & 0x20) && (flags | 0x0);
}

void normal_with_boolean() {
    bool b = true;
    bool r;
    r = b & 0xFFFFFFF8;
}

// The case is taken from the real code in clang-tools-extra/clangd/CodeComplete.cpp
bool test_invalid_fixit_flag_check(unsigned Flags) {
    bool Deprecated = true;
    bool bool1 = true, bool2 = false;
    Deprecated = Deprecated && Flags & 0xFFFFFFF8;
    Deprecated = (Flags & 0xFFFFFFF8) && Deprecated;

    Deprecated = Deprecated & Flags & 0xFFFFFFF8;
    Deprecated = (Flags & 0xFFFFFFF8) & Deprecated;

    Deprecated &= Flags & 0xFFFFFFF8;
    Deprecated &= 0xFFFFFFF8 & Flags;

    Deprecated = 0xFFFFFFF8 & (bool1 & bool2);
    // CHECK-FIXES: Deprecated = 0xFFFFFFF8 & (bool1 && bool2);
    Deprecated = (bool1 & bool2) & 0xFFFFFFF8;
    // CHECK-FIXES: Deprecated = (bool1 && bool2) & 0xFFFFFFF8;

    return Deprecated;
}

bool test_invalid_fixit_flag_check_with_or(unsigned Flags) {
    bool Deprecated = true;
    bool bool1 = true, bool2 = false;
    Deprecated = Deprecated || (Flags | 0xFFFFFFF8);
    Deprecated = (Flags | 0xFFFFFFF8) || Deprecated;

    Deprecated = Deprecated | (Flags | 0xFFFFFFF8);
    Deprecated = (Flags | 0xFFFFFFF8) | Deprecated;

    Deprecated |= Flags | 0xFFFFFFF8;
    Deprecated |= 0xFFFFFFF8 | Flags;

    Deprecated = 0xFFFFFFF8 | (bool1 | bool2);
    // CHECK-FIXES: Deprecated = 0xFFFFFFF8 | (bool1 || bool2);
    Deprecated = (bool1 | bool2) | 0xFFFFFFF8;
    // CHECK-FIXES: Deprecated = (bool1 || bool2) | 0xFFFFFFF8;

    return Deprecated;
}

bool test_invalid_fixit_flag_check_with_and_or(unsigned Flags) {
    bool Deprecated = true;
    bool bool1 = true, bool2 = false;
    Deprecated = Deprecated && ((Flags & 0xFFFFFFF8) | 0x20);
    Deprecated = ((Flags & 0xFFFFFFF8) | 0x20) && Deprecated;

    Deprecated = Deprecated & ((Flags & 0xFFFFFFF8) | 0x20);
    Deprecated = ((Flags & 0xFFFFFFF8) | 0x20) & Deprecated;

    Deprecated &= (Flags & 0xFFFFFFF8) | 0x20;
    Deprecated &= 0x20 | (Flags & 0xFFFFFFF8);

    Deprecated = 0xFFFFFFF8 & (bool1 | bool2);
    // CHECK-FIXES: Deprecated = 0xFFFFFFF8 & (bool1 || bool2);
    Deprecated = (bool1 | bool2) & 0xFFFFFFF8;
    // CHECK-FIXES: Deprecated = (bool1 || bool2) & 0xFFFFFFF8;

    return Deprecated;
}

bool test_invalid_fixit_flag_check_long_chains(unsigned Flags, bool Bool1, bool Bool2, bool Bool3, bool Bool4, bool Bool5) {
    bool Deprecated = true;

    // Long chain of booleans with one flag operation in the middle
    Deprecated = Bool1 & Bool2 & (Flags & 0xFFFFFFF8) & Bool3 & Bool4;
    // CHECK-FIXES: Deprecated = Bool1 && Bool2 & (Flags & 0xFFFFFFF8) & Bool3 & Bool4;

    // Long chain of booleans with two flag operations
    Deprecated = Bool1 & (Flags & 0xFFFFFFF8) & Bool2 & (Flags & 0x20) & Bool3;

    // Long chain starting with boolean, flag operation in the middle
    Deprecated = Deprecated & Bool1 & Bool2 & (Flags & 0xFFFFFFF8) & Bool3 & Bool4;
    // CHECK-FIXES: Deprecated = Deprecated && Bool1 && Bool2 & (Flags & 0xFFFFFFF8) & Bool3 & Bool4;

    // Very long chain with flag operation near the end
    Deprecated = Bool1 & Bool2 & Bool3 & Bool4 & (Flags & 0xFFFFFFF8) & Bool5;
    // CHECK-FIXES: Deprecated = Bool1 && Bool2 && Bool3 && Bool4 & (Flags & 0xFFFFFFF8) & Bool5;

    // Long chain with flag operation at the beginning (after first boolean)
    Deprecated = Bool1 & (Flags & 0xFFFFFFF8) & Bool2 & Bool3 & Bool4 & Bool5;

    // Long chain with compound assignment and flag operation
    Deprecated &= Bool1 & Bool2 & (Flags & 0xFFFFFFF8) & Bool3;
    // CHECK-FIXES: Deprecated &= Bool1 && Bool2 & (Flags & 0xFFFFFFF8) & Bool3;

    // Long chain with two consecutive flag operations
    Deprecated = Bool1 & (Flags & 0xFFFFFFF8) & (Flags & 0x20) & Bool2 & Bool3;

    Deprecated = (Flags & 0xFFFFFFF8) & Bool1 & (Flags & 0x20) & Bool2 & Bool3;

    // Same tests as above, but with reversed order of evaluation.

    // Long chain of booleans with one flag operation in the middle (reversed)
    Deprecated = Bool4 & Bool3 & (Flags & 0xFFFFFFF8) & Bool2 & Bool1;
    // CHECK-FIXES: Deprecated = Bool4 && Bool3 & (Flags & 0xFFFFFFF8) & Bool2 & Bool1;

    // Long chain of booleans with two flag operations (reversed)
    Deprecated = Bool3 & (Flags & 0x20) & Bool2 & (Flags & 0xFFFFFFF8) & Bool1;

    // Long chain starting with boolean, flag operation in the middle (reversed)
    Deprecated = Bool4 & Bool3 & (Flags & 0xFFFFFFF8) & Bool2 & Bool1 & Deprecated;
    // CHECK-FIXES: Deprecated = Bool4 && Bool3 & (Flags & 0xFFFFFFF8) & Bool2 & Bool1 & Deprecated;

    // Very long chain with flag operation near the end (reversed)
    Deprecated = Bool5 & (Flags & 0xFFFFFFF8) & Bool4 & Bool3 & Bool2 & Bool1;

    // Long chain with flag operation at the beginning (after first boolean) (reversed)
    Deprecated = Bool5 & Bool4 & Bool3 & Bool2 & (Flags & 0xFFFFFFF8) & Bool1;
    // CHECK-FIXES: Deprecated = Bool5 && Bool4 && Bool3 && Bool2 & (Flags & 0xFFFFFFF8) & Bool1;

    // Long chain with compound assignment and flag operation (reversed)
    Deprecated &= Bool3 & (Flags & 0xFFFFFFF8) & Bool2 & Bool1;

    // Long chain with two consecutive flag operations (reversed)
    Deprecated = Bool3 & Bool2 & (Flags & 0x20) & (Flags & 0xFFFFFFF8) & Bool1;
    // CHECK-FIXES: Deprecated = Bool3 && Bool2 & (Flags & 0x20) & (Flags & 0xFFFFFFF8) & Bool1;

    Deprecated = Bool3 & Bool2 & (Flags & 0x20) & Bool1 & (Flags & 0xFFFFFFF8);
    // CHECK-FIXES: Deprecated = Bool3 && Bool2 & (Flags & 0x20) & Bool1 & (Flags & 0xFFFFFFF8);

    return Deprecated;
}

bool test_invalid_fixit_flag_check_long_chains_with_or(unsigned Flags, bool Bool1, bool Bool2, bool Bool3, bool Bool4, bool Bool5) {
    bool Deprecated = true;

    // Long chain of booleans with one flag operation in the middle
    Deprecated = Bool1 | Bool2 | (Flags | 0xFFFFFFF8) | Bool3 | Bool4;
    // CHECK-FIXES: Deprecated = Bool1 || Bool2 | (Flags | 0xFFFFFFF8) | Bool3 | Bool4;

    // Long chain of booleans with two flag operations
    Deprecated = Bool1 | (Flags | 0xFFFFFFF8) | Bool2 | (Flags | 0x20) | Bool3;

    // Long chain starting with boolean, flag operation in the middle
    Deprecated = Deprecated | Bool1 | Bool2 | (Flags | 0xFFFFFFF8) | Bool3 | Bool4;
    // CHECK-FIXES: Deprecated = Deprecated || Bool1 || Bool2 | (Flags | 0xFFFFFFF8) | Bool3 | Bool4;

    // Very long chain with flag operation near the end
    Deprecated = Bool1 | Bool2 | Bool3 | Bool4 | (Flags | 0xFFFFFFF8) | Bool5;
    // CHECK-FIXES: Deprecated = Bool1 || Bool2 || Bool3 || Bool4 | (Flags | 0xFFFFFFF8) | Bool5;

    // Long chain with flag operation at the beginning (after first boolean)
    Deprecated = Bool1 | (Flags | 0xFFFFFFF8) | Bool2 | Bool3 | Bool4 | Bool5;

    // Long chain with compound assignment and flag operation
    Deprecated |= Bool1 | Bool2 | (Flags | 0xFFFFFFF8) | Bool3;
    // CHECK-FIXES: Deprecated |= Bool1 || Bool2 | (Flags | 0xFFFFFFF8) | Bool3;

    // Long chain with two consecutive flag operations
    Deprecated = Bool1 | (Flags | 0xFFFFFFF8) | (Flags | 0x20) | Bool2 | Bool3;

    Deprecated = (Flags | 0xFFFFFFF8) | Bool1 | (Flags | 0x20) | Bool2 | Bool3;

    // Same tests as above, but with reversed order of evaluation.

    // Long chain of booleans with one flag operation in the middle (reversed)
    Deprecated = Bool4 | Bool3 | (Flags | 0xFFFFFFF8) | Bool2 | Bool1;
    // CHECK-FIXES: Deprecated = Bool4 || Bool3 | (Flags | 0xFFFFFFF8) | Bool2 | Bool1;

    // Long chain of booleans with two flag operations (reversed)
    Deprecated = Bool3 | (Flags | 0x20) | Bool2 | (Flags | 0xFFFFFFF8) | Bool1;

    // Long chain starting with boolean, flag operation in the middle (reversed)
    Deprecated = Bool4 | Bool3 | (Flags | 0xFFFFFFF8) | Bool2 | Bool1 | Deprecated;
    // CHECK-FIXES: Deprecated = Bool4 || Bool3 | (Flags | 0xFFFFFFF8) | Bool2 | Bool1 | Deprecated;

    // Very long chain with flag operation near the end (reversed)
    Deprecated = Bool5 | (Flags | 0xFFFFFFF8) | Bool4 | Bool3 | Bool2 | Bool1;

    // Long chain with flag operation at the beginning (after first boolean) (reversed)
    Deprecated = Bool5 | Bool4 | Bool3 | Bool2 | (Flags | 0xFFFFFFF8) | Bool1;
    // CHECK-FIXES: Deprecated = Bool5 || Bool4 || Bool3 || Bool2 | (Flags | 0xFFFFFFF8) | Bool1;

    // Long chain with compound assignment and flag operation (reversed)
    Deprecated |= Bool3 | (Flags | 0xFFFFFFF8) | Bool2 | Bool1;

    // Long chain with two consecutive flag operations (reversed)
    Deprecated = Bool3 | Bool2 | (Flags | 0x20) | (Flags | 0xFFFFFFF8) | Bool1;
    // CHECK-FIXES: Deprecated = Bool3 || Bool2 | (Flags | 0x20) | (Flags | 0xFFFFFFF8) | Bool1;

    Deprecated = Bool3 | Bool2 | (Flags | 0x20) | Bool1 | (Flags | 0xFFFFFFF8);
    // CHECK-FIXES: Deprecated = Bool3 || Bool2 | (Flags | 0x20) | Bool1 | (Flags | 0xFFFFFFF8);

    return Deprecated;
}

bool test_invalid_fixit_flag_check_long_chains_mixed(unsigned Flags, bool Bool1, bool Bool2, bool Bool3, bool Bool4, bool Bool5) {
    bool Deprecated = true;

    // Mixed operators with one flag operation in the middle
    Deprecated = Bool1 & Bool2 | (Flags & 0xFFFFFFF8) & Bool3 & Bool4;
    // CHECK-FIXES: Deprecated = (Bool1 && Bool2) | (Flags & 0xFFFFFFF8) & Bool3 & Bool4;

    // Mixed operators with two flag operations
    Deprecated = Bool1 & Bool2 | (Flags & 0xFFFFFFF8) & Bool3 | (Flags & 0x20) & Bool4;
    // CHECK-FIXES: Deprecated = (Bool1 && Bool2) | (Flags & 0xFFFFFFF8) & Bool3 | (Flags & 0x20) & Bool4;

    // Mixed operators where boolean sub-expression uses '|'
    Deprecated = (Bool1 | Bool2) & (Flags & 0xFFFFFFF8) & Bool3 & Bool4;
    // CHECK-FIXES: Deprecated = (Bool1 || Bool2) & (Flags & 0xFFFFFFF8) & Bool3 & Bool4;

    // Long chain starting with boolean, flag operation in the middle and mixed operators
    Deprecated = Deprecated & Bool1 & Bool2 | (Flags & 0xFFFFFFF8) | (Bool3 & Bool4);
    // CHECK-FIXES: Deprecated = (Deprecated && Bool1 && Bool2) | (Flags & 0xFFFFFFF8) | (Bool3 && Bool4);

    // Long chain with compound assignment and mixed operators
    Deprecated &= Bool1 & Bool2 | (Flags & 0xFFFFFFF8) & Bool3;
    // CHECK-FIXES: Deprecated &= (Bool1 && Bool2) | (Flags & 0xFFFFFFF8) & Bool3;

    // Same patterns as above, but with reversed order of evaluation.

    Deprecated = Bool4 & Bool3 | (Flags & 0xFFFFFFF8) & Bool2 & Bool1;
    // CHECK-FIXES: Deprecated = (Bool4 && Bool3) | (Flags & 0xFFFFFFF8) & Bool2 & Bool1;

    Deprecated = (Bool4 | Bool3) & (Flags & 0xFFFFFFF8) & Bool2 & Bool1;
    // CHECK-FIXES: Deprecated = (Bool4 || Bool3) & (Flags & 0xFFFFFFF8) & Bool2 & Bool1;

    Deprecated &= Bool3 & Bool2 | (Flags & 0xFFFFFFF8) & Bool1;
    // CHECK-FIXES: Deprecated &= (Bool3 && Bool2) | (Flags & 0xFFFFFFF8) & Bool1;

    return Deprecated;
}
