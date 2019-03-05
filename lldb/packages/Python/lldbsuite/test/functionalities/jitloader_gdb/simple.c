#include <inttypes.h>

// GDB JIT interface stub
struct
{
    uint32_t version;
    uint32_t action_flag;
    void* relevant_entry;
    void* first_entry;
} __jit_debug_descriptor = { 1, 0, 0, 0 };

void __jit_debug_register_code()
{
}
// end GDB JIT interface stub

int main()
{
    return 0;
}
