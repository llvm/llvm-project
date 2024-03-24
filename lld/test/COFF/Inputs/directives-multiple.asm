; fasm2 directives-multiple.asm
; lld-link /dll directives-multiple.obj

format MS64 COFF

; NOTE: Each function has its own .drectve section

section '.text' code readable executable align 16

; BOOL WINAPI
; DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved)
DllMain:
	mov eax, 1 ; TRUE
	retn

public DllMain as '_DllMainCRTStartup' ; linker expects this default name



section '.text' code readable executable align 16
align 16
_Add:	; __declspec(dllexport) int Add(int a, int b) {
	lea eax, [rcx + rdx]
	retn

public _Add as '?Add@@YAHHH@Z' ; decorated name for linker

section '.drectve' linkinfo linkremove
db "/EXPORT:Add=?Add@@YAHHH@Z "



section '.text' code readable executable align 16
align 16
_Sub:	; __declspec(dllexport) int Subtract(int a, int b) {
	xchg eax, ecx
	sub eax, edx
	retn

public _Sub as '?Subtract@@YAHHH@Z' ; decorated name for linker

section '.drectve' linkinfo linkremove
db "/EXPORT:Subtract=?Subtract@@YAHHH@Z "
