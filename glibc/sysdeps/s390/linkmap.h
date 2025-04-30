#if __WORDSIZE == 64
struct link_map_machine
  {
    Elf64_Addr plt; /* Address of .plt + 0x2e */
    const Elf64_Rela *jmprel; /* Address of first JMP_SLOT reloc */
  };
#else
struct link_map_machine
  {
    Elf32_Addr plt; /* Address of .plt + 0x2c */
    const Elf32_Rela *jmprel; /* Address of first JMP_SLOT reloc */
  };
#endif
