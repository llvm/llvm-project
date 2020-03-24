#include "cilksan_internal.h"
#include "shadow_mem.h"
#include "shadow_mem_allocator.h"
#include "simple_shadow_mem.h"
// #include "hash_shadow_mem.h"
// #include "compresseddict_shadow_mem.h"
// #include "noop_shadow_mem.h"

extern CilkSanImpl_t CilkSanImpl;

// Initialize custom memory allocators for dictionaries in shadow memory.
template<>
MALineAllocator &SimpleDictionary<0>::MAAlloc =
  CilkSanImpl.getMALineAllocator(0);
template<>
MALineAllocator &SimpleDictionary<1>::MAAlloc =
  CilkSanImpl.getMALineAllocator(1);
template<>
MALineAllocator &SimpleDictionary<2>::MAAlloc =
  CilkSanImpl.getMALineAllocator(2);

void Shadow_Memory::init()
{
  type = 0;
    // std::cout << "shadow memory type is: " << type << std::endl;
    // switch(type) {
    //     // case 0: shadow_mem = new Simple_Shadow_Memory(); break;
    //     // case 1: shadow_mem = new Hash_Shadow_Memory(); break;
    //     case 2: shadow_mem = new CompressedDictShadowMem(); break;
    //     // case 3: shadow_mem = new Noop_Shadow_Memory(); break;
    // }
  shadow_mem = new SimpleShadowMem();
}
