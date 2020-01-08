#include <mram.h>
#include <stdint.h>

int main() {
        uint64_t buffer;
        mram_read8(0, &buffer);
        if (buffer == 0xfabddbafdeadbeefULL)
                return 0;
        else
                return -1;
}
