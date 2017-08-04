// RUN: %clang_cc1 %s -cl-std=CL2.0 -triple "spir-unknown-unknown" -verify -pedantic -fsyntax-only -DB32 -DQUALS=
// RUN: %clang_cc1 %s -cl-std=CL2.0 -triple "spir-unknown-unknown" -verify -pedantic -fsyntax-only -DB32 -DQUALS="const volatile"
// RUN: %clang_cc1 %s -cl-std=CL2.0 -triple "spir64-unknown-unknown" -verify -pedantic -fsyntax-only -Wconversion -DWCONV -DQUALS=
// RUN: %clang_cc1 %s -cl-std=CL2.0 -triple "spir64-unknown-unknown" -verify -pedantic -fsyntax-only -Wconversion -DWCONV -DQUALS="const volatile"

typedef struct {int a;} ndrange_t;
// Diagnostic tests for different overloads of enqueue_kernel from Table 6.13.17.1 of OpenCL 2.0 Spec.
kernel void enqueue_kernel_tests() {
  queue_t default_queue;
  unsigned flags = 0;
  QUALS ndrange_t ndrange;
  clk_event_t evt;
  clk_event_t event_wait_list;
  clk_event_t event_wait_list2[] = {evt, evt};
  void *vptr;

  // Testing the first overload type
  enqueue_kernel(default_queue, flags, ndrange, ^(void) {
    return 0;
  });

  enqueue_kernel(vptr, flags, ndrange, ^(void) { // expected-error{{illegal call to 'enqueue_kernel', expected 'queue_t' argument type}}
    return 0;
  });

  enqueue_kernel(default_queue, vptr, ndrange, ^(void) { // expected-error{{illegal call to 'enqueue_kernel', expected 'kernel_enqueue_flags_t' (i.e. uint) argument type}}
    return 0;
  });

  enqueue_kernel(default_queue, flags, vptr, ^(void) { // expected-error{{illegal call to 'enqueue_kernel', expected 'ndrange_t' argument type}}
    return 0;
  });

  enqueue_kernel(default_queue, flags, ndrange, vptr); // expected-error{{illegal call to 'enqueue_kernel', expected block argument}}

  enqueue_kernel(default_queue, flags, ndrange, ^(int i) { // expected-error{{blocks with parameters are not accepted in this prototype of enqueue_kernel call}}
    return 0;
  });

  // Testing the second overload type
  enqueue_kernel(default_queue, flags, ndrange, 1, &event_wait_list, &evt, ^(void) {
                                                                             return 0;
                                                                           });

  enqueue_kernel(default_queue, flags, ndrange, 1, 0, 0, ^(void) {
                                                           return 0;
                                                         });

  enqueue_kernel(default_queue, flags, ndrange, vptr, &event_wait_list, &evt, ^(void) { // expected-error{{illegal call to 'enqueue_kernel', expected integer argument type}}
    return 0;
  });

  enqueue_kernel(default_queue, flags, ndrange, 1, vptr, &evt, ^(void) // expected-error{{illegal call to 'enqueue_kernel', expected 'clk_event_t *' argument type}}
                                                               {
                                                                 return 0;
                                                               });

  enqueue_kernel(default_queue, flags, ndrange, 1, &event_wait_list, vptr, ^(void) // expected-error{{illegal call to 'enqueue_kernel', expected 'clk_event_t *' argument type}}
                                                                           {
                                                                             return 0;
                                                                           });

  enqueue_kernel(default_queue, flags, ndrange, 1, &event_wait_list, &evt, vptr); // expected-error{{illegal call to 'enqueue_kernel', expected block argument}}

  // Testing the third overload type
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 1024, 1024);

  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 1024L, 1024);

  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 1024, 4294967296L);
#ifdef B32
// expected-warning@-2{{implicit conversion from 'long' to 'unsigned int' changes value from 4294967296 to 0}}
#endif

  char c;
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 c, 1024L);
#ifdef WCONV
// expected-warning-re@-2{{implicit conversion changes signedness: 'char' to 'unsigned {{int|long}}'}}
#endif
#define UINT_MAX 4294967295

  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 sizeof(int), sizeof(int) * UINT_MAX);
#ifdef B32
// expected-warning@-2{{implicit conversion from 'long' to 'unsigned int' changes value from 17179869180 to 4294967292}}
#endif

  typedef void (^bl_A_t)(local void *);

  const bl_A_t block_A = (bl_A_t) ^ (local void *a) {};

  enqueue_kernel(default_queue, flags, ndrange, block_A, 1024);

  typedef void (^bl_B_t)(local void *, local int *);

  const bl_B_t block_B = (bl_B_t) ^ (local void *a, local int *b) {};

  enqueue_kernel(default_queue, flags, ndrange, block_B, 1024, 1024); // expected-error{{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}

  enqueue_kernel(default_queue, flags, ndrange, // expected-error{{mismatch in number of block parameters and local size arguments passed}}
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 1024);

  float illegal_mem_size = (float)0.5f;
  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 illegal_mem_size, illegal_mem_size); // expected-error{{illegal call to enqueue_kernel, parameter needs to be specified as integer type}} expected-error{{illegal call to enqueue_kernel, parameter needs to be specified as integer type}}

  enqueue_kernel(default_queue, flags, ndrange,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 illegal_mem_size, 1024); // expected-error{{illegal call to enqueue_kernel, parameter needs to be specified as integer type}}

  // Testing the forth overload type
  enqueue_kernel(default_queue, flags, ndrange, 1, event_wait_list2, &evt,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 1024, 1024);

  enqueue_kernel(default_queue, flags, ndrange, 1, 0, 0,
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 1024, 1024);

  enqueue_kernel(default_queue, flags, ndrange, 1, &event_wait_list, &evt, // expected-error{{mismatch in number of block parameters and local size arguments passed}}
                 ^(local void *a, local void *b) {
                   return 0;
                 },
                 1024, 1024, 1024);

  // More random misc cases that can't be deduced
  enqueue_kernel(default_queue, flags, ndrange, 1, &event_wait_list, &evt); // expected-error{{illegal call to enqueue_kernel, incorrect argument types}}

  enqueue_kernel(default_queue, flags, ndrange, 1, 1); // expected-error{{illegal call to enqueue_kernel, incorrect argument types}}
}

// Diagnostic tests for get_kernel_work_group_size and allowed block parameter types in dynamic parallelism.
kernel void work_group_size_tests() {
  void (^const block_A)(void) = ^{
    return;
  };
  void (^const block_B)(int) = ^(int a) {
    return;
  };
  void (^const block_C)(local void *) = ^(local void *a) {
    return;
  };
  void (^const block_D)(local int *) = ^(local int *a) {
    return;
  };

  unsigned size = get_kernel_work_group_size(block_A);
  size = get_kernel_work_group_size(block_C);
  size = get_kernel_work_group_size(^(local void *a) {
    return;
  });
  size = get_kernel_work_group_size(^(local int *a) { // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
    return;
  });
  size = get_kernel_work_group_size(block_B);   // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
  size = get_kernel_work_group_size(block_D);   // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
  size = get_kernel_work_group_size(^(int a) {  // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
    return;
  });
  size = get_kernel_work_group_size();          // expected-error {{too few arguments to function call, expected 1, have 0}}
  size = get_kernel_work_group_size(1);         // expected-error{{expected block argument}}
  size = get_kernel_work_group_size(block_A, 1); // expected-error{{too many arguments to function call, expected 1, have 2}}

  size = get_kernel_preferred_work_group_size_multiple(block_A);
  size = get_kernel_preferred_work_group_size_multiple(block_C);
  size = get_kernel_preferred_work_group_size_multiple(^(local void *a) {
    return;
  });
  size = get_kernel_preferred_work_group_size_multiple(^(local int *a) { // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
    return;
  });
  size = get_kernel_preferred_work_group_size_multiple(^(int a) {  // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
    return;
  });
  size = get_kernel_preferred_work_group_size_multiple(block_B);   // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
  size = get_kernel_preferred_work_group_size_multiple(block_D);   // expected-error {{blocks used in enqueue_kernel call are expected to have parameters of type 'local void*'}}
  size = get_kernel_preferred_work_group_size_multiple();          // expected-error {{too few arguments to function call, expected 1, have 0}}
  size = get_kernel_preferred_work_group_size_multiple(1);         // expected-error{{expected block argument}}
  size = get_kernel_preferred_work_group_size_multiple(block_A, 1); // expected-error{{too many arguments to function call, expected 1, have 2}}
}
