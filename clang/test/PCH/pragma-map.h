// Header for PCH test pragma-map.c

// None of these symbols is declared in this header; the declarations come from
// the file that includes it.
#pragma map(undecfunc1, "UNDEC1")
#pragma map(undecfunc2, "UNDEC2")
#pragma map(undecfunc3, "UNDEC3")
#pragma map(undecfunc4, "UNDEC4")
#pragma map(undecfunc5, "UNDEC5")
#pragma map(undecfunc6, "UNDEC6")
#pragma map(undecvar, "UNDECVAR")

// A numeric escape sequence is an IBM-1047 code point; '@' is 0x7C (\174).
#pragma map(undecesc, "\174ESC")

// The symbol is declared after the pragma, in this header.
#pragma map(declfunc, "DECL")
void declfunc(void);
