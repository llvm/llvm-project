struct link_map_machine
  {
    Elf32_Addr plt; /* Address of .plt */
    void *tlsdesc_table; /* Address of TLS descriptor hash table.  */
  };
