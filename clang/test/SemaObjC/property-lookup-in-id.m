// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://9106929

typedef struct objc_class *Class;

typedef struct objc_object {
    Class isa;
} *id;


typedef struct __FSEventStream* FSEventStreamRef;

extern id NSApp;

@interface FileSystemMonitor { 

 FSEventStreamRef fsEventStream;
}
@property(assign) FSEventStreamRef fsEventStream;

@end

@implementation FileSystemMonitor
@synthesize fsEventStream;

- (void)startFSEventGathering:(id)sender
{
  fsEventStream = [NSApp delegate].fsEventStream; // expected-warning {{instance method '-delegate' not found (return type defaults to 'id')}} \
                                                  // expected-error {{property 'fsEventStream' not found on object of type 'id'}}

}
@end

