#include <mram.h>
#include <stdint.h>

int main() {
        uint64_t buffer;
        mram_read8(0xffffffff, &buffer);
        return (int)buffer;
}
