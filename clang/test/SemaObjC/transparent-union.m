// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

typedef union {
 struct xx_object_s *_do;
 struct xx_continuation_s *_dc;
 struct xx_queue_s *_dq;
 struct xx_queue_attr_s *_dqa;
 struct xx_group_s *_dg;
 struct xx_source_s *_ds;
 struct xx_source_attr_s *_dsa;
 struct xx_semaphore_s *_dsema;
} xx_object_t __attribute__((transparent_union));

@interface INTF
- (void) doSomething : (xx_object_t) xxObject;
- (void)testMeth;
@end

@implementation INTF
- (void) doSomething : (xx_object_t) xxObject {}
- (void)testMeth { struct xx_queue_s *sq; [self doSomething:sq ]; }
@end
