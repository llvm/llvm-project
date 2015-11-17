int foo __attribute__((section(".gcc_except_table.foo"))) = 4;
const int bar __attribute__((section(".data.rel.local"))) = 2;
const int baz __attribute__((section(".data.rel.ro"))) = 2;
const int bak __attribute__((section(".data.xyz"))) = 2;
