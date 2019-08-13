#define TASKLETS_INITIALIZER TASKLET( main, 1024, 0)
#include <rt.h>

#include <mram.h>
#include <stdint.h>

int main() {
        uint64_t buffer;
        mram_read8(0xffffffff, &buffer);
        return (int)buffer;
}
