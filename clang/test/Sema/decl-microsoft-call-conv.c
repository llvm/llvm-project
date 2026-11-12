// RUN: %clang_cc1 -triple i686-pc-win32 -Wno-strict-prototypes -verify %s

// It's important that this is a .c file.

// This is fine, as CrcGenerateTable*() has a prototype.
void __fastcall CrcGenerateTableFastcall(void);
void __fastcall CrcGenerateTableFastcall();
void __fastcall CrcGenerateTableFastcall() {}
void __stdcall CrcGenerateTableStdcall(void);
void __stdcall CrcGenerateTableStdcall();
void __stdcall CrcGenerateTableStdcall() {}
void __thiscall CrcGenerateTableThiscall(void);
void __thiscall CrcGenerateTableThiscall();
void __thiscall CrcGenerateTableThiscall() {}
void __pascal CrcGenerateTablePascal(void);
void __pascal CrcGenerateTablePascal();
void __pascal CrcGenerateTablePascal() {}
void __vectorcall CrcGenerateTableVectorcall(void);
void __vectorcall CrcGenerateTableVectorcall();
void __vectorcall CrcGenerateTableVectorcall() {}

void __fastcall CrcGenerateTableNoProtoFastcall(); // expected-error{{function with no prototype cannot use the fastcall calling convention}}
void __stdcall CrcGenerateTableNoProtoStdcall(); // expected-warning{{function with no prototype cannot use the stdcall calling convention}}
void __thiscall CrcGenerateTableNoProtoThiscall(); // expected-error{{function with no prototype cannot use the thiscall calling convention}}
void __pascal CrcGenerateTableNoProtoPascal(); // expected-error{{function with no prototype cannot use the pascal calling convention}}
void __vectorcall CrcGenerateTableNoProtoVectorcall(); // expected-error{{function with no prototype cannot use the vectorcall calling convention}}

void __fastcall CrcGenerateTableNoProtoDefFastcall() {}
void __stdcall CrcGenerateTableNoProtoDefStdcall() {}
void __thiscall CrcGenerateTableNoProtoDefThiscall() {}
void __pascal CrcGenerateTableNoProtoDefPascal() {}
void __vectorcall CrcGenerateTableNoProtoDefVectorcall() {}

// Regular calling convention is fine.
void CrcGenerateTableNoProto() {}
