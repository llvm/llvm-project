#include "shadow_mem.h"
// #include "simple_shadow_mem.h"
// #include "hash_shadow_mem.h"
#include "compresseddict_shadow_mem.h"
// #include "noop_shadow_mem.h"

void Shadow_Memory::init()
{
    type = 2;
    std::cout << "shadow memory type is: " << type << std::endl;
    switch(type) {
        // case 0: shadow_mem = new Simple_Shadow_Memory(); break;
        // case 1: shadow_mem = new Hash_Shadow_Memory(); break;
        case 2: shadow_mem = new CompressedDictShadowMem(); break;
        // case 3: shadow_mem = new Noop_Shadow_Memory(); break;
    }
}
