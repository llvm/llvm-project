void func() {
	__asm__ (
		"pushq $0x10;"
		".cfi_def_cfa_offset 16;"
		"jmp label;"
		"movq $0x48, %rax;"
"label: subq $0x38, %rax;"
		"movq $0x48, %rcx;"
		"movq $0x48, %rdx;"
		"movq $0x48, %rax;"
		"popq %rax;"
	);

}


int main(int argc, char const *argv[])
{
	func();
}