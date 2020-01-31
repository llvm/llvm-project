#include <mram.h>
#include <stdint.h>

int main() {
        uint64_t buffer;
        mram_read(0xffffffff, &buffer, sizeof(uint64_t));
        return (int)buffer;
}
