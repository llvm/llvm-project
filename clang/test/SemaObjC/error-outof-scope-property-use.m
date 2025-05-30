// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class %s

@class NSMutableDictionary;

@interface LaunchdJobs 

@property (nonatomic,retain) NSMutableDictionary *uuids_jobs;

@end

@implementation LaunchdJobs

-(void)job
{

 [uuids_jobs objectForKey]; // expected-error {{use of undeclared identifier 'uuids_jobs'}}
}


@end

void
doLaunchdJobCPU(void)
{
 [uuids_jobs enumerateKeysAndObjectsUsingBlock]; // expected-error {{use of undeclared identifier 'uuids_jobs'}}
}
