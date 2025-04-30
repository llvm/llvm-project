struct link_map_machine
  {
    ElfW(Addr) plt; /* Address of .plt */
    ElfW(Word) fpabi; /* FP ABI of the object */
    unsigned int odd_spreg; /* Does the object require odd_spreg support? */
    const Elf32_Word *mips_xlat_zero; /* .MIPS.xhash */
  };
