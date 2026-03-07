// RUN: %check_clang_tidy %s misc-bool-bitwise-operation %t

// The case is taken from the real code in clang/lib/APINotes/APINotesWriter.cpp

void take(bool value) {}
void take_int(int value) {}

void general(unsigned flags, bool value) {
    (void)((flags << 1) | value);
    flags = (flags << 1) | value;
    flags = (flags << 1) | (flags << 2) | value;
    flags = (flags << 1) | (flags << 2) | (flags << 4) | value;
    take_int((flags << 1) | value);
    take_int((flags << 1) | (flags << 2) | value);
    (void)(value | (flags << 1));
    flags = value | (flags << 1);
    flags = value | (flags << 1) | (flags << 2);
    flags = value | (flags << 1) | (flags << 2) | (flags << 4);
    flags = (flags << 1) | value | (flags << 2);
    flags = (flags << 1) | value | (flags << 2) | (flags << 4);
    flags = (flags << 1) | (flags << 2) | value | (flags << 4);
    take_int(value | (flags << 1));
    take_int(value | (flags << 1) | (flags << 2));
    take_int((flags << 1) | value | (flags << 2));
}

// FIXME: implement `template<bool bb=true|1>` cases

void assign_to_boolean(unsigned flags, bool value) {
    // FIXME: this is false negative, fix it
    // struct A { bool a = true | 1; };
    // struct B { union { bool a = true | 1; }; };
}

void assign_to_boolean_parens(unsigned flags, bool value) {
    // FIXME: this is false negative, fix it
    // struct A { bool a = (true | 1); };
    // struct B { union { bool a = (true | 1); }; };
}

void assign_to_boolean_parens2(unsigned flags, bool value) {
    // FIXME: this is false negative, fix it
    // struct A { bool a = ((true | 1)); };
    // struct B { union { bool a = ((true | 1)); }; };
}

// functional cast
void assign_to_boolean_fcast(unsigned flags, bool value) {
    // FIXME: this is false negative, fix it
    // struct A { bool a = bool(true | 1); };
    // struct B { union { bool a = bool(true | 1); }; };
}

// C-style cast
void assign_to_boolean_ccast(unsigned flags, bool value) {
    // FIXME: this is false negative, fix it
    // struct A { bool a = (bool)(true | 1); };
    // struct B { union { bool a = (bool)(true | 1); }; };
}

// static_cast
void assign_to_boolean_scast(unsigned flags, bool value) {
    // FIXME: this is false negative, fix it
    // struct A { bool a = static_cast<bool>(true | 1); };
    // struct B { union { bool a = static_cast<bool>(true | 1); }; };
}


