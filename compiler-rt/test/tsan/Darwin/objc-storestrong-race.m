// RUN: %clang_tsan %s -fobjc-arc -o %t-ww -framework Foundation
// RUN: %deflake %run %t-ww 2>&1 | FileCheck %s --check-prefixes=CHECK,WW
// RUN: %clang_tsan %s -fobjc-arc -DREAD_WRITE -o %t-rw -framework Foundation
// RUN: %deflake %run %t-rw 2>&1 | FileCheck %s --check-prefixes=CHECK,RW

#import "../test.h"
#import <Foundation/Foundation.h>
#import <pthread.h>

@interface MyClass : NSObject
@property(strong, nonatomic) NSString *dangerous;
@end

@implementation MyClass
@end

MyClass *c;

void *Thread1(void *x) {
  barrier_wait(&barrier);
#ifdef READ_WRITE
  NSString *s = c.dangerous;
  (void)s;
#else
  c.dangerous = [NSString stringWithFormat:@"thread1"];
#endif
  return NULL;
}

void *Thread2(void *x) {
  barrier_wait(&barrier);
  c.dangerous = [NSString stringWithFormat:@"thread2"];
  return NULL;
}

int main(int argc, char *argv[]) {
  fprintf(stderr, "Hello world.\n");

  c = [[MyClass alloc] init];
  c.dangerous = [NSString stringWithFormat:@"main"];

  barrier_init(&barrier, 2);
  pthread_t t[2];
  pthread_create(&t[0], NULL, Thread1, NULL);
  pthread_create(&t[1], NULL, Thread2, NULL);
  pthread_join(t[0], NULL);
  pthread_join(t[1], NULL);

  fprintf(stderr, "Done.\n");
}

// CHECK: Hello world.
// CHECK: WARNING: ThreadSanitizer: data race

// WW: Write of size 8 at [[ADDR:0x[0-9a-f]+]]
// WW: -[MyClass setDangerous:]
// WW: Previous write of size 8 at [[ADDR]]
// WW: -[MyClass setDangerous:]

// RW: {{Read|Write}} of size 8 at [[ADDR:0x[0-9a-f]+]]
// RW: -[MyClass {{dangerous|setDangerous:}}]
// RW: Previous {{read|write}} of size 8 at [[ADDR]]
// RW: -[MyClass {{dangerous|setDangerous:}}]

// CHECK: Done.
