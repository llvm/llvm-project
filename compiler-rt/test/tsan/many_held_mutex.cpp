// RUN: %clangxx_tsan %s -fsanitize=thread -o %t && %run %t 2>&1 | Filecheck %s

#include <mutex>
#include <stdio.h>

int main(){
    const unsigned short NUM_OF_MTX = 128;
    std::mutex mutexes[NUM_OF_MTX];

    for(int i = 0; i < NUM_OF_MTX; i++){
        mutexes[i].lock();
    }
    for(int i = 0; i < NUM_OF_MTX; i++){
        mutexes[i].unlock();
    }

    printf("Success\n");

    return 0;
}

// CHECK: Success
// CHECK-NOT: ThreadSanitizer: CHECK failed
