// RUN: %clang_cc1 -verify -fsyntax-only -fblocks -fobjc-exceptions -Wcompletion-handler %s

#define NULL (void *)0
#define nil (id)0
#define CALLED_ONCE __attribute__((called_once))
#define NORETURN __attribute__((noreturn))

@protocol NSObject
@end
@interface NSObject <NSObject>
- (id)copy;
- (id)class;
- autorelease;
@end

typedef unsigned int NSUInteger;
typedef struct {
} NSFastEnumerationState;

@interface NSArray <__covariant NSFastEnumeration>
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@interface NSMutableArray<ObjectType> : NSArray <ObjectType>
- addObject:anObject;
@end
@class NSString, Protocol;
extern void NSLog(NSString *format, ...);

void escape(void (^callback)(void));
void escape_void(void *);
void indirect_call(void (^callback)(void) CALLED_ONCE);
void indirect_conv(void (^completionHandler)(void));
void filler(void);
void exit(int) NORETURN;

void double_call_one_block(void (^callback)(void) CALLED_ONCE) {
  callback(); // expected-note{{previous call is here}}
  callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void double_call_one_block_parens(void (^callback)(void) CALLED_ONCE) {
  (callback)(); // expected-note{{previous call is here}}
  (callback)(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void double_call_one_block_ptr(void (*callback)(void) CALLED_ONCE) {
  callback(); // expected-note{{previous call is here}}
  callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void double_call_one_block_ptr_deref(void (*callback)(void) CALLED_ONCE) {
  (*callback)(); // expected-note{{previous call is here}}
  (*callback)(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void multiple_call_one_block(void (^callback)(void) CALLED_ONCE) {
  // We don't really need to repeat the same warning for the same parameter.
  callback(); // no-warning
  callback(); // no-warning
  callback(); // no-warning
  callback(); // expected-note{{previous call is here}}
  callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void double_call_branching_1(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    callback(); // expected-note{{previous call is here}}
  } else {
    cond += 42;
  }
  callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void double_call_branching_2(int cond, void (^callback)(void) CALLED_ONCE) {
  callback();
  // expected-note@-1{{previous call is here; set to nil to indicate it cannot be called afterwards}}

  if (cond) {
    callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
  } else {
    cond += 42;
  }
}

void double_call_branching_3(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    callback();
  } else {
    callback();
  }
  // no-warning
}

void double_call_branching_4(int cond1, int cond2, void (^callback)(void) CALLED_ONCE) {
  if (cond1) {
    cond2 = !cond2;
  } else {
    callback();
    // expected-note@-1{{previous call is here; set to nil to indicate it cannot be called afterwards}}
  }

  if (cond2) {
    callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
  }
}

void double_call_loop(int counter, void (^callback)(void) CALLED_ONCE) {
  while (counter > 0) {
    counter--;
    // Both note and warning are on the same line, which is a common situation
    // in loops.
    callback(); // expected-note{{previous call is here}}
    // expected-warning@-1{{'callback' parameter marked 'called_once' is called twice}}
  }
}

void never_called_trivial(void (^callback)(void) CALLED_ONCE) {
  // expected-warning@-1{{'callback' parameter marked 'called_once' is never called}}
}

int never_called_branching(int x, void (^callback)(void) CALLED_ONCE) {
  // expected-warning@-1{{'callback' parameter marked 'called_once' is never called}}
  x -= 42;

  if (x == 10) {
    return 0;
  }

  return x + 15;
}

void escaped_one_block_1(void (^callback)(void) CALLED_ONCE) {
  escape(callback); // no-warning
}

void escaped_one_block_2(void (^callback)(void) CALLED_ONCE) {
  escape(callback); // no-warning
  callback();
}

void escaped_one_path_1(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    escape(callback); // no-warning
  } else {
    callback();
  }
}

void escaped_one_path_2(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    escape(callback); // no-warning
  }

  callback();
}

void escaped_one_path_3(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    // expected-warning@-1{{'callback' parameter marked 'called_once' is never used when taking false branch}}
    escape(callback);
  }
}

void escape_in_between_1(void (^callback)(void) CALLED_ONCE) {
  callback(); // expected-note{{previous call is here}}
  escape(callback);
  callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void escape_in_between_2(int cond, void (^callback)(void) CALLED_ONCE) {
  callback(); // expected-note{{previous call is here}}
  if (cond) {
    escape(callback);
  }
  callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void escape_in_between_3(int cond, void (^callback)(void) CALLED_ONCE) {
  callback(); // expected-note{{previous call is here}}

  if (cond) {
    escape(callback);
  } else {
    escape_void((__bridge void *)callback);
  }

  callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void escaped_as_void_ptr(void (^callback)(void) CALLED_ONCE) {
  escape_void((__bridge void *)callback); // no-warning
}

void indirect_call_no_warning_1(void (^callback)(void) CALLED_ONCE) {
  indirect_call(callback); // no-warning
}

void indirect_call_no_warning_2(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    indirect_call(callback);
  } else {
    callback();
  }
  // no-warning
}

void indirect_call_double_call(void (^callback)(void) CALLED_ONCE) {
  indirect_call(callback); // expected-note{{previous call is here}}
  callback();              // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

void indirect_call_within_direct_call(void (^callback)(void) CALLED_ONCE,
                                      void (^meta)(void (^param)(void) CALLED_ONCE) CALLED_ONCE) {
  // TODO: Report warning for 'callback'.
  //       At the moment, it is not possible to access 'called_once' attribute from the type
  //       alone when there is no actual declaration of the marked parameter.
  meta(callback);
  callback();
  // no-warning
}

void block_call_1(void (^callback)(void) CALLED_ONCE) {
  indirect_call(^{
    callback();
  });
  callback();
  // no-warning
}

void block_call_2(void (^callback)(void) CALLED_ONCE) {
  escape(^{
    callback();
  });
  callback();
  // no-warning
}

void block_call_3(int cond, void (^callback)(void) CALLED_ONCE) {
  ^{
    if (cond) {
      callback(); // expected-note{{previous call is here}}
    }
    callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
  }();          // no-warning
}

void block_call_4(int cond, void (^callback)(void) CALLED_ONCE) {
  ^{
    if (cond) {
      // expected-warning@-1{{'callback' parameter marked 'called_once' is never used when taking false branch}}
      escape(callback);
    }
  }();
}

void block_call_5(void (^outer)(void) CALLED_ONCE) {
  ^(void (^inner)(void) CALLED_ONCE) {
    // expected-warning@-1{{'inner' parameter marked 'called_once' is never called}}
  }(outer);
}

void block_with_called_once(void (^outer)(void) CALLED_ONCE) {
  escape_void((__bridge void *)^(void (^inner)(void) CALLED_ONCE) {
    inner(); // expected-note{{previous call is here}}
    inner(); // expected-warning{{'inner' parameter marked 'called_once' is called twice}}
  });
  outer(); // expected-note{{previous call is here}}
  outer(); // expected-warning{{'outer' parameter marked 'called_once' is called twice}}
}

void never_called_one_exit(int cond, void (^callback)(void) CALLED_ONCE) {
  if (!cond) // expected-warning{{'callback' parameter marked 'called_once' is never called when taking true branch}}
    return;

  callback();
}

void never_called_if_then_1(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) { // expected-warning{{'callback' parameter marked 'called_once' is never called when taking true branch}}
  } else {
    callback();
  }
}

void never_called_if_then_2(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) { // expected-warning{{'callback' parameter marked 'called_once' is never called when taking true branch}}
    // This way the first statement in the basic block is different from
    // the first statement in the compound statement
    (void)cond;
  } else {
    callback();
  }
}

void never_called_if_else_1(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) { // expected-warning{{'callback' parameter marked 'called_once' is never called when taking false branch}}
    callback();
  } else {
  }
}

void never_called_if_else_2(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) { // expected-warning{{'callback' parameter marked 'called_once' is never called when taking false branch}}
    callback();
  }
}

void never_called_two_ifs(int cond1, int cond2, void (^callback)(void) CALLED_ONCE) {
  if (cond1) {   // expected-warning{{'callback' parameter marked 'called_once' is never called when taking false branch}}
    if (cond2) { // expected-warning{{'callback' parameter marked 'called_once' is never called when taking true branch}}
      return;
    }
    callback();
  }
}

void never_called_ternary_then(int cond, void (^other)(void), void (^callback)(void) CALLED_ONCE) {
  return cond ? // expected-warning{{'callback' parameter marked 'called_once' is never called when taking true branch}}
             other()
              : callback();
}

void never_called_for_false(int size, void (^callback)(void) CALLED_ONCE) {
  for (int i = 0; i < size; ++i) {
    // expected-warning@-1{{'callback' parameter marked 'called_once' is never called when skipping the loop}}
    callback();
    break;
  }
}

void never_called_for_true(int size, void (^callback)(void) CALLED_ONCE) {
  for (int i = 0; i < size; ++i) {
    // expected-warning@-1{{'callback' parameter marked 'called_once' is never called when entering the loop}}
    return;
  }
  callback();
}

void never_called_while_false(int cond, void (^callback)(void) CALLED_ONCE) {
  while (cond) { // expected-warning{{'callback' parameter marked 'called_once' is never called when skipping the loop}}
    callback();
    break;
  }
}

void never_called_while_true(int cond, void (^callback)(void) CALLED_ONCE) {
  while (cond) { // expected-warning{{'callback' parameter marked 'called_once' is never called when entering the loop}}
    return;
  }
  callback();
}

void never_called_switch_case(int cond, void (^callback)(void) CALLED_ONCE) {
  switch (cond) {
  case 1:
    callback();
    break;
  case 2:
    callback();
    break;
  case 3: // expected-warning{{'callback' parameter marked 'called_once' is never called when handling this case}}
    break;
  default:
    callback();
    break;
  }
}

void never_called_switch_default(int cond, void (^callback)(void) CALLED_ONCE) {
  switch (cond) {
  case 1:
    callback();
    break;
  case 2:
    callback();
    break;
  default: // expected-warning{{'callback' parameter marked 'called_once' is never called when handling this case}}
    break;
  }
}

void never_called_switch_two_cases(int cond, void (^callback)(void) CALLED_ONCE) {
  switch (cond) {
  case 1: // expected-warning{{'callback' parameter marked 'called_once' is never called when handling this case}}
    break;
  case 2: // expected-warning{{'callback' parameter marked 'called_once' is never called when handling this case}}
    break;
  default:
    callback();
    break;
  }
}

void never_called_switch_none(int cond, void (^callback)(void) CALLED_ONCE) {
  switch (cond) { // expected-warning{{'callback' parameter marked 'called_once' is never called when none of the cases applies}}
  case 1:
    callback();
    break;
  case 2:
    callback();
    break;
  }
}

enum YesNoOrMaybe {
  YES,
  NO,
  MAYBE
};

void exhaustive_switch(enum YesNoOrMaybe cond, void (^callback)(void) CALLED_ONCE) {
  switch (cond) {
  case YES:
    callback();
    break;
  case NO:
    callback();
    break;
  case MAYBE:
    callback();
    break;
  }
  // no-warning
}

void called_twice_exceptions(void (^callback)(void) CALLED_ONCE) {
  // TODO: Obj-C exceptions are not supported in CFG,
  //       we should report warnings in these as well.
  @try {
    callback();
    callback();
  }
  @finally {
    callback();
  }
}

void noreturn_1(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    exit(1);
  } else {
    callback();
  }
  // no-warning
}

void noreturn_2(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    callback();
    exit(1);
  } else {
    callback();
  }
  // no-warning
}

void noreturn_3(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    exit(1);
  }

  callback();
  // no-warning
}

void noreturn_4(void (^callback)(void) CALLED_ONCE) {
  exit(1);
  // no-warning
}

void noreturn_5(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    // NOTE: This is an ambiguous case caused by the fact that we do a backward
    //       analysis.  We can probably report it here, but for the sake of
    //       the simplicity of our analysis, we don't.
    if (cond == 42) {
      callback();
    }
    exit(1);
  }
  callback();
  // no-warning
}

void never_called_noreturn_1(int cond, void (^callback)(void) CALLED_ONCE) {
  // expected-warning@-1{{'callback' parameter marked 'called_once' is never called}}
  if (cond) {
    exit(1);
  }
}

void double_call_noreturn(int cond, void (^callback)(void) CALLED_ONCE) {
  callback(); // expected-note{{previous call is here}}

  if (cond) {
    if (cond == 42) {
      callback(); // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
    }
    exit(1);
  }
}

void call_with_check_1(void (^callback)(void) CALLED_ONCE) {
  if (callback)
    callback();
  // no-warning
}

void call_with_check_2(void (^callback)(void) CALLED_ONCE) {
  if (!callback) {
  } else {
    callback();
  }
  // no-warning
}

void call_with_check_3(void (^callback)(void) CALLED_ONCE) {
  if (callback != NULL)
    callback();
  // no-warning
}

void call_with_check_4(void (^callback)(void) CALLED_ONCE) {
  if (NULL != callback)
    callback();
  // no-warning
}

void call_with_check_5(void (^callback)(void) CALLED_ONCE) {
  if (callback == NULL) {
  } else {
    callback();
  }
  // no-warning
}

void call_with_check_6(void (^callback)(void) CALLED_ONCE) {
  if (NULL == callback) {
  } else {
    callback();
  }
  // no-warning
}

int call_with_check_7(int (^callback)(void) CALLED_ONCE) {
  return callback ? callback() : 0;
  // no-warning
}

void unreachable_true_branch(void (^callback)(void) CALLED_ONCE) {
  if (0) {

  } else {
    callback();
  }
  // no-warning
}

void unreachable_false_branch(void (^callback)(void) CALLED_ONCE) {
  if (1) {
    callback();
  }
  // no-warning
}

void never_called_conv_1(void (^completionHandler)(void)) {
  // expected-warning@-1{{completion handler is never called}}
}

void never_called_conv_2(void (^completion)(void)) {
  // expected-warning@-1{{completion handler is never called}}
}

void never_called_conv_WithCompletion(void (^callback)(void)) {
  // expected-warning@-1{{completion handler is never called}}
}

void indirectly_called_conv(void (^completionHandler)(void)) {
  indirect_conv(completionHandler);
  // no-warning
}

void escape_through_assignment_1(void (^callback)(void) CALLED_ONCE) {
  id escapee;
  escapee = callback;
  escape(escapee);
  // no-warning
}

void escape_through_assignment_2(void (^callback)(void) CALLED_ONCE) {
  id escapee = callback;
  escape(escapee);
  // no-warning
}

void escape_through_assignment_3(void (^callback1)(void) CALLED_ONCE,
                                 void (^callback2)(void) CALLED_ONCE) {
  id escapee1 = callback1, escapee2 = callback2;
  escape(escapee1);
  escape(escapee2);
  // no-warning
}

void not_called_in_throw_branch_1(id exception, void (^callback)(void) CALLED_ONCE) {
  if (exception) {
    @throw exception;
  }

  callback();
}

void not_called_in_throw_branch_2(id exception, void (^callback)(void) CALLED_ONCE) {
  // expected-warning@-1{{'callback' parameter marked 'called_once' is never called}}
  if (exception) {
    @throw exception;
  }
}

void conventional_error_path_1(int error, void (^completionHandler)(void)) {
  if (error) {
    // expected-warning@-1{{completion handler is never called when taking true branch}}
    // This behavior might be tweaked in the future
    return;
  }

  completionHandler();
}

void conventional_error_path_2(int error, void (^callback)(void) CALLED_ONCE) {
  // Conventions do not apply to explicitly marked parameters.
  if (error) {
    // expected-warning@-1{{'callback' parameter marked 'called_once' is never called when taking true branch}}
    return;
  }

  callback();
}

void suppression_1(void (^callback)(void) CALLED_ONCE) {
  // This is a way to tell the analysis that we know about this path,
  // and we do not want to call the callback here.
  (void)callback; // no-warning
}

void suppression_2(int cond, void (^callback)(void) CALLED_ONCE) {
  if (cond) {
    (void)callback; // no-warning
  } else {
    callback();
  }
}

void suppression_3(int cond, void (^callback)(void) CALLED_ONCE) {
  // Even if we do this on one of the paths, it doesn't mean we should
  // forget about other paths.
  if (cond) {
    // expected-warning@-1{{'callback' parameter marked 'called_once' is never used when taking false branch}}
    (void)callback;
  }
}

@interface TestBase : NSObject
- (void)escape:(void (^)(void))callback;
- (void)indirect_call:(void (^)(void))CALLED_ONCE callback;
- (void)indirect_call_conv_1:(int)cond
           completionHandler:(void (^)(void))completionHandler;
- (void)indirect_call_conv_2:(int)cond
           completionHandler:(void (^)(void))handler;
- (void)indirect_call_conv_3WithCompletion:(void (^)(void))handler;
- (void)indirect_call_conv_4:(void (^)(void))handler
    __attribute__((swift_async(swift_private, 1)));
- (void)exit:(int)code NORETURN;
- (int)condition;
@end

@interface TestClass : TestBase
@property(strong) NSMutableArray *handlers;
@property(strong) id storedHandler;
@property int wasCanceled;
@property(getter=hasErrors) int error;
@end

@implementation TestClass

- (void)double_indirect_call_1:(void (^)(void))CALLED_ONCE callback {
  [self indirect_call:callback]; // expected-note{{previous call is here}}
  [self indirect_call:callback]; // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
}

- (void)double_indirect_call_2:(void (^)(void))CALLED_ONCE callback {
  [self indirect_call_conv_1:0 // expected-note{{previous call is here}}
           completionHandler:callback];
  [self indirect_call_conv_1:1 // expected-warning{{'callback' parameter marked 'called_once' is called twice}}
           completionHandler:callback];
}

- (void)double_indirect_call_3:(void (^)(void))completionHandler {
  [self indirect_call_conv_2:0 // expected-note{{previous call is here}}
           completionHandler:completionHandler];
  [self indirect_call_conv_2:1 // expected-warning{{completion handler is called twice}}
           completionHandler:completionHandler];
}

- (void)double_indirect_call_4:(void (^)(void))completion {
  [self indirect_call_conv_2:0 // expected-note{{previous call is here}}
           completionHandler:completion];
  [self indirect_call_conv_2:1 // expected-warning{{completion handler is called twice}}
           completionHandler:completion];
}

- (void)double_indirect_call_5:(void (^)(void))withCompletionHandler {
  [self indirect_call_conv_2:0 // expected-note{{previous call is here}}
           completionHandler:withCompletionHandler];
  [self indirect_call_conv_2:1 // expected-warning{{completion handler is called twice}}
           completionHandler:withCompletionHandler];
}

- (void)double_indirect_call_6:(void (^)(void))completionHandler {
  [self indirect_call_conv_3WithCompletion: // expected-note{{previous call is here}}
            completionHandler];
  [self indirect_call_conv_3WithCompletion: // expected-warning{{completion handler is called twice}}
            completionHandler];
}

- (void)double_indirect_call_7:(void (^)(void))completionHandler {
  [self indirect_call_conv_4: // expected-note{{previous call is here}}
            completionHandler];
  [self indirect_call_conv_4: // expected-warning{{completion handler is called twice}}
            completionHandler];
}

- (void)never_called_trivial:(void (^)(void))CALLED_ONCE callback {
  // expected-warning@-1{{'callback' parameter marked 'called_once' is never called}}
  filler();
}

- (void)noreturn:(int)cond callback:(void (^)(void))CALLED_ONCE callback {
  if (cond) {
    [self exit:1];
  }

  callback();
  // no-warning
}

- (void)escaped_one_path:(int)cond callback:(void (^)(void))CALLED_ONCE callback {
  if (cond) {
    [self escape:callback]; // no-warning
  } else {
    callback();
  }
}

- (void)block_call_1:(void (^)(void))CALLED_ONCE callback {
  // We consider captures by blocks as escapes
  [self indirect_call:(^{
          callback();
        })];
  callback();
  // no-warning
}

- (void)block_call_2:(int)cond callback:(void (^)(void))CALLED_ONCE callback {
  [self indirect_call:
            ^{
              if (cond) {
                // expected-warning@-1{{'callback' parameter marked 'called_once' is never used when taking false branch}}
                [self escape:callback];
              }
            }];
}

- (void)block_call_3:(int)cond
    completionHandler:(void (^)(void))callback {
  [self indirect_call:
            ^{
              if (cond) {
                // expected-warning@-1{{completion handler is never used when taking false branch}}
                [self escape:callback];
              }
            }];
}

- (void)block_call_4WithCompletion:(void (^)(void))callback {
  [self indirect_call:
            ^{
              if ([self condition]) {
                // expected-warning@-1{{completion handler is never used when taking false branch}}
                [self escape:callback];
              }
            }];
}

- (void)never_called_conv:(void (^)(void))completionHandler {
  // expected-warning@-1{{completion handler is never called}}
  filler();
}

- (void)indirectly_called_conv:(void (^)(void))completionHandler {
  indirect_conv(completionHandler);
  // no-warning
}

- (void)never_called_one_exit_conv:(int)cond completionHandler:(void (^)(void))handler {
  if (!cond) // expected-warning{{completion handler is never called when taking true branch}}
    return;

  handler();
}

- (void)escape_through_assignment:(void (^)(void))completionHandler {
  _storedHandler = completionHandler;
  // no-warning
}

- (void)escape_through_copy:(void (^)(void))completionHandler {
  _storedHandler = [completionHandler copy];
  // no-warning
}

- (void)escape_through_copy_and_autorelease:(void (^)(void))completionHandler {
  _storedHandler = [[completionHandler copy] autorelease];
  // no-warning
}

- (void)complex_escape:(void (^)(void))completionHandler {
  if (completionHandler) {
    [_handlers addObject:[[completionHandler copy] autorelease]];
  }
  // no-warning
}

- (void)test_crash:(void (^)(void))completionHandler cond:(int)cond {
  if (cond) {
    // expected-warning@-1{{completion handler is never used when taking false branch}}
    for (id _ in _handlers) {
    }

    [_handlers addObject:completionHandler];
  }
}

- (void)conventional_error_path_1:(void (^)(void))completionHandler {
  if (self.wasCanceled)
    // expected-warning@-1{{completion handler is never called when taking true branch}}
    // This behavior might be tweaked in the future
    return;

  completionHandler();
}

- (void)conventional_error_path_2:(void (^)(void))completionHandler {
  if (self.wasCanceled)
    // expected-warning@-1{{completion handler is never used when taking true branch}}
    // This behavior might be tweaked in the future
    return;

  [_handlers addObject:completionHandler];
}

- (void)conventional_error_path_3:(void (^)(void))completionHandler {
  if (self.hasErrors)
    // expected-warning@-1{{completion handler is never called when taking true branch}}
    // This behavior might be tweaked in the future
    return;

  completionHandler();
}

- (void)conventional_error_path_3:(int)cond completionHandler:(void (^)(void))handler {
  if (self.wasCanceled)
    // expected-warning@-1{{completion handler is never called when taking true branch}}
    // TODO: When we have an error on some other path, in order not to prevent it from
    //       being reported, we report this one as well.
    //       Probably, we should address this at some point.
    return;

  if (cond) {
    // expected-warning@-1{{completion handler is never called when taking false branch}}
    handler();
  }
}

#define NSAssert(condition, desc, ...) NSLog(desc, ##__VA_ARGS__);

- (void)empty_base_1:(void (^)(void))completionHandler {
  NSAssert(0, @"Subclass must implement");
  // no-warning
}

- (void)empty_base_2:(void (^)(void))completionHandler {
  // no-warning
}

- (int)empty_base_3:(void (^)(void))completionHandler {
  return 1;
  // no-warning
}

- (int)empty_base_4:(void (^)(void))completionHandler {
  NSAssert(0, @"Subclass must implement");
  return 1;
  // no-warning
}

- (int)empty_base_5:(void (^)(void))completionHandler {
  NSAssert(0, @"%@ doesn't support", [self class]);
  return 1;
  // no-warning
}

#undef NSAssert
#define NSAssert(condition, desc, ...) \
  if (!(condition)) {                  \
    NSLog(desc, ##__VA_ARGS__);        \
  }

- (int)empty_base_6:(void (^)(void))completionHandler {
  NSAssert(0, @"%@ doesn't support", [self class]);
  return 1;
  // no-warning
}

#undef NSAssert
#define NSAssert(condition, desc, ...) \
  do {                                 \
    NSLog(desc, ##__VA_ARGS__);        \
  } while (0)

- (int)empty_base_7:(void (^)(void))completionHandler {
  NSAssert(0, @"%@ doesn't support", [self class]);
  return 1;
  // no-warning
}

- (void)two_conditions_1:(int)first
                  second:(int)second
       completionHandler:(void (^)(void))completionHandler {
  if (first && second) {
    // expected-warning@-1{{completion handler is never called when taking false branch}}
    completionHandler();
  }
}

- (void)two_conditions_2:(int)first
                  second:(int)second
       completionHandler:(void (^)(void))completionHandler {
  if (first || second) {
    // expected-warning@-1{{completion handler is never called when taking true branch}}
    return;
  }

  completionHandler();
}

- (void)testWithCompletionHandler:(void (^)(void))callback {
  if ([self condition]) {
    // expected-warning@-1{{completion handler is never called when taking false branch}}
    callback();
  }
}

- (void)testWithCompletion:(void (^)(void))callback {
  if ([self condition]) {
    // expected-warning@-1{{completion handler is never called when taking false branch}}
    callback();
  }
}

- (void)completion_handler_wrong_type:(int (^)(void))completionHandler {
  // We don't want to consider completion handlers with non-void return types.
  if ([self condition]) {
    // no-warning
    completionHandler();
  }
}

- (void)test_swift_async_none:(int)cond
            completionHandler:(void (^)(void))handler __attribute__((swift_async(none))) {
  if (cond) {
    // no-warning
    handler();
  }
}

- (void)test_swift_async_param:(int)cond
                      callback:(void (^)(void))callback
    __attribute__((swift_async(swift_private, 2))) {
  if (cond) {
    // expected-warning@-1{{completion handler is never called when taking false branch}}
    callback();
  }
}

- (void)test_nil_suggestion:(int)cond1
                     second:(int)cond2
                 completion:(void (^)(void))handler {
  if (cond1) {
    handler();
    // expected-note@-1{{previous call is here; set to nil to indicate it cannot be called afterwards}}
  }

  if (cond2) {
    handler(); // expected-warning{{completion handler is called twice}}
  }
}

- (void)test_nil_suppression_1:(int)cond1
                        second:(int)cond2
                    completion:(void (^)(void))handler {
  if (cond1) {
    handler();
    handler = nil;
    // no-warning
  }

  if (cond2) {
    handler();
  }
}

- (void)test_nil_suppression_2:(int)cond1
                        second:(int)cond2
                    completion:(void (^)(void))handler {
  if (cond1) {
    handler();
    handler = NULL;
    // no-warning
  }

  if (cond2) {
    handler();
  }
}

- (void)test_nil_suppression_3:(int)cond1
                        second:(int)cond2
                    completion:(void (^)(void))handler {
  if (cond1) {
    handler();
    handler = 0;
    // no-warning
  }

  if (cond2) {
    handler();
  }
}

@end
