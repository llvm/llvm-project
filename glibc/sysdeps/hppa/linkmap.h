/* Used to store the function descriptor table */
struct link_map_machine
  {
    size_t fptr_table_len;
    ElfW(Addr) *fptr_table;
  };
