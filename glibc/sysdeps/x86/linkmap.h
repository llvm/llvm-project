#if __WORDSIZE == 64
struct link_map_machine
  {
    Elf64_Addr plt; /* Address of .plt + 0x16 */
    Elf64_Addr gotplt; /* Address of .got + 0x18 */
    void *tlsdesc_table; /* Address of TLS descriptor hash table.  */
  };

#else
struct link_map_machine
  {
    Elf32_Addr plt; /* Address of .plt + 0x16 */
    Elf32_Addr gotplt; /* Address of .got + 0x0c */
    void *tlsdesc_table; /* Address of TLS descriptor hash table.  */
  };
#endif
