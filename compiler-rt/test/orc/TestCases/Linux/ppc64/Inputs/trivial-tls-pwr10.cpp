extern thread_local int x;
extern thread_local int y;
extern thread_local int z;

int __attribute__((target("arch=pwr10"))) TestPOWER10() { return x + y + z; }
