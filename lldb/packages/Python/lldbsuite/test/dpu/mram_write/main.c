#include <mram.h>
#include <stdint.h>

int main() {
        uint64_t buffer;
        mram_read(0, &buffer, sizeof(buffer));
        if (buffer == 0xfabddbafdeadbeefULL)
                return 0;
        else
                return -1;
}
