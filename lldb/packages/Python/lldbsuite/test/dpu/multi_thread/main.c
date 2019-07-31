#define NB_THREAD 16

#define TASKLETS_INITIALIZER TASKLETS(NB_THREAD, main, 1024, 0)
#include <rt.h>

#include <defs.h>

__attribute__((noinline)) int f_rec(int loop, int it, int val){
        if (!loop)
                return val;
        for (unsigned int i = 0; i < loop; i++) {
                val = (val << it) + it;
        }
        it = (it + NB_THREAD - 1) % NB_THREAD;
        return f_rec(loop - 1, it, val); // Step location 2
}

int main() {
        const sysname_t tid = me(); // Breakpoint location
        return f_rec(8, tid, tid); // Step location 1
}
