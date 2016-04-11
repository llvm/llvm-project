//===-- main.c --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>
#import <pthread.h>

long my_global;

void *Thread1(void *arg) {
    my_global = 42;
    return NULL;
}

void *Thread2(void *arg) {
    my_global = 144;
    return NULL;
}

void TestDataRace1() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, Thread1, NULL);
    pthread_create(&t2, NULL, Thread2, NULL);
    
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
}

void TestInvalidMutex() {
    pthread_mutex_t m = {0};
    pthread_mutex_lock(&m);
    
    pthread_mutex_init(&m, NULL);
    pthread_mutex_lock(&m);
    pthread_mutex_unlock(&m);
    pthread_mutex_destroy(&m);
    pthread_mutex_lock(&m);
}

void TestMutexWrongLock() {
    pthread_mutex_t m = {0};
    pthread_mutex_init(&m, NULL);
    pthread_mutex_unlock(&m);
}

long some_global;

void TestDataRaceBlocks1() {
    dispatch_queue_t q = dispatch_queue_create("my.queue", DISPATCH_QUEUE_CONCURRENT);
    
    for (int i = 0; i < 2; i++) {
        dispatch_async(q, ^{
            some_global++;  // race 1
            
            usleep(100000);  // force the blocks to be on different threads
        });
    }
    
    usleep(100000);
    dispatch_barrier_sync(q, ^{ });
}

void TestDataRaceBlocks2() {
    dispatch_queue_t q = dispatch_queue_create("my.queue2", DISPATCH_QUEUE_CONCURRENT);
    
    char *c;
    
    c = malloc((rand() % 1000) + 10);
    for (int i = 0; i < 2; i++) {
        dispatch_async(q, ^{
            c[0] = 'x';  // race 2
            fprintf(stderr, "tid: %p\n", pthread_self());
            usleep(100000);  // force the blocks to be on different threads
        });
    }
    dispatch_barrier_sync(q, ^{ });
    
    free(c);
}

void TestUseAfterFree() {
    char *c;
    
    c = malloc((rand() % 1000) + 10);
    free(c);
    c[0] = 'x';
}

void TestRacePipe() {
    dispatch_queue_t q = dispatch_queue_create("my.queue3", DISPATCH_QUEUE_CONCURRENT);
    
    int a[2];
    pipe(a);
    int fd = a[0];
    
    for (int i = 0; i < 2; i++) {
        dispatch_async(q, ^{
            write(fd, "abc", 3);
            usleep(100000);  // force the blocks to be on different threads
        });
        dispatch_async(q, ^{
            close(fd);
            usleep(100000);
        });
    }
    
    dispatch_barrier_sync(q, ^{ });
}

void TestThreadLeak() {
    pthread_t t1;
    pthread_create(&t1, NULL, Thread1, NULL);
}

int main(int argc, const char * argv[]) {
    TestDataRace1();
    
    TestInvalidMutex();
    
    TestMutexWrongLock();
    
    TestDataRaceBlocks1();
    
    TestDataRaceBlocks2();
    
    TestUseAfterFree();
    
    TestRacePipe();
    
    TestThreadLeak();

    return 0;
}
