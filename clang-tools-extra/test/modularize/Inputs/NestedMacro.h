// Verification of fix for nested macro.

#define FUNCMACROINNER(a) a
#define FUNCMACROOUTER(b, c) FUNCMACROINNER(b) + FUNCMACROINNER(c)
int FuncMacroValue = FUNCMACROOUTER(1, 2);
